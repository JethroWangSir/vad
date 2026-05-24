import os
import glob
import time
import csv
import torch
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_score, recall_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from thop import profile, clever_format
from tqdm import tqdm

# 匯入您更新後的 SGVAD 類別
from sgvad import SGVAD

def get_vad_score(vad_model, wave_path):
    device = vad_model.cfg.device  # 直接使用設定檔中的裝置
    
    try:
        wave = vad_model.load_audio(wave_path)
    except Exception as e:
        tqdm.write(f"\n[警告] 無法讀取音檔 {wave_path}: {e}")
        return None, 0.0, 0.0

    # 檢查 1：是否為空音檔
    if len(wave) == 0:
        tqdm.write(f"\n[警告] 略過空音檔: {wave_path}")
        return None, 0.0, 0.0
        
    # 檢查 2：是否短於 STFT 視窗大小 (n_fft 預設為 512)
    if len(wave) < vad_model.cfg.preprocessor.n_fft:
        tqdm.write(f"\n[警告] 略過過短的音檔 (少於 {vad_model.cfg.preprocessor.n_fft} samples): {wave_path}")
        return None, 0.0, 0.0

    # 將音訊張量搬移至指定的裝置 (與 Preprocessor 保持一致)
    wave = torch.tensor(wave, dtype=torch.float32).reshape(1, -1).to(device)
    wave_len = torch.tensor([wave.size(-1)], dtype=torch.long).reshape(1).to(device)
    
    processed_signal, processed_signal_len = vad_model.preprocessor(input_signal=wave, length=wave_len)
    
    # 若使用 GPU，需同步以確保計時精準
    if "cuda" in device:
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    with torch.no_grad():
        mu, _ = vad_model.model(audio_signal=processed_signal, length=processed_signal_len)
        binary_gates = torch.clamp(mu + 0.5, 0.0, 1.0)
        score = binary_gates.sum(dim=1).mean().item()
        
    if "cuda" in device:
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    audio_duration_sec = wave.size(-1) / vad_model.cfg.sample_rate
    inference_time = end_time - start_time
    
    return score, audio_duration_sec, inference_time

class NeMoModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio_signal, length):
        return self.model(audio_signal=audio_signal, length=length)

def profile_model(vad_model):
    """計算模型的參數量與 MACs"""
    print("\n--- 模型複雜度分析 ---")
    device = vad_model.cfg.device
    total_params = sum(p.numel() for p in vad_model.model.parameters())
    print(f"==> 總參數量 (Parameters): {total_params / 1e3:.2f} k")

    dummy_frames = 300
    # 確保虛擬輸入也在正確的裝置上
    dummy_signal = torch.randn(1, vad_model.cfg.preprocessor.n_mels, dummy_frames).to(device)
    dummy_len = torch.tensor([dummy_frames], dtype=torch.long).to(device)
    
    try:
        wrapper = NeMoModelWrapper(vad_model.model)
        macs, params = profile(wrapper, inputs=(dummy_signal, dummy_len), verbose=False)
        macs_str, params_str = clever_format([macs, params], "%.2f")
        print(f"==> 運算量 (MACs, 基準為 3 秒音訊): {macs_str}")
    except Exception as e:
        print(f"無法計算 MACs，錯誤訊息: {e}")
    print("----------------------\n")

def main():
    # ==========================================
    # 參數設定區
    # ==========================================
    base_dir = "/share/nas169/jethrowang/SincQDR-VAD/data/AVA/dataset" 
    output_dir = "./eval_ava/official"
    
    # 確保輸出資料夾存在
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, "metrics_report.txt")
    csv_file = os.path.join(output_dir, "predictions.csv")

    print("正在初始化模型...")
    vad_model = SGVAD.init_from_ckpt()
    print(f"模型初始化完成 (當前使用裝置: {vad_model.cfg.device})")
    
    profile_model(vad_model)
    
    category_to_label = {
        "NO_SPEECH": 0,
        "CLEAN_SPEECH": 1,
        "SPEECH_WITH_NOISE": 1,
        "SPEECH_WITH_MUSIC": 1
    }
    
    y_true = []
    y_scores = []
    file_paths = []
    total_audio_duration = 0.0
    total_inference_time = 0.0
    
    print("開始處理音檔與推論...\n")
    for category, label in category_to_label.items():
        folder_path = os.path.join(base_dir, category)
        wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
        if not wav_files: 
            continue
            
        for fpath in tqdm(wav_files, desc=f"處理 [{category}]", unit="file"):
            score, duration, inf_time = get_vad_score(vad_model, fpath)
            
            if score is None:
                continue
                
            y_scores.append(score)
            y_true.append(label)
            file_paths.append(fpath)
            total_audio_duration += duration
            total_inference_time += inf_time

    if not y_true:
        print("未找到資料。")
        return

    # 計算指標
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    roc_auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    target_fpr = 0.315
    idx = np.argmin(np.abs(fpr - target_fpr))
    tpr_at_fpr = tpr[idx]

    cfg_threshold = vad_model.cfg.threshold
    y_pred = (y_scores >= cfg_threshold).astype(int)
    rtf = total_inference_time / total_audio_duration

    # 準備報告字串
    report_str = f"""=============================================
SGVAD AVA 資料集測試報告
=============================================
測試總音檔數: {len(y_true)}
總音檔時長: {total_audio_duration:.2f} 秒
總推論耗時: {total_inference_time:.2f} 秒 (僅計算 Model Forward)
硬體裝置: {vad_model.cfg.device}

--- 效能評估指標 (Performance Metrics) ---
==> ROCAUC: {roc_auc:.4f}
==> EER: {eer * 100:.2f}%
==> TPR @ FPR={target_fpr}: {tpr_at_fpr:.4f} (論文參考值約為 0.96)

--- 實際決策指標 (Threshold = {cfg_threshold}) ---
==> F1-Score:  {f1_score(y_true, y_pred):.4f}
==> Precision: {precision_score(y_true, y_pred):.4f}
==> Recall:    {recall_score(y_true, y_pred):.4f}

--- 速度指標 (Speed Metrics) ---
==> Real-Time Factor (RTF): {rtf:.4f} (小於 1 代表可即時處理)
=============================================
"""

    # 1. 輸出至終端機
    print(f"\n{report_str}")
    
    # 2. 儲存報告至 TXT
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"[儲存成功] 指標報告已儲存至: {report_file}")

    # 3. 儲存原始預測資料至 CSV
    with open(csv_file, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "true_label", "predicted_score"])
        for path, true_val, score in zip(file_paths, y_true, y_scores):
            writer.writerow([path, true_val, score])
    print(f"[儲存成功] 原始預測結果已儲存至: {csv_file}")

if __name__ == "__main__":
    main()
