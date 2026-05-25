import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from nemo.collections.asr.models import EncDecClassificationModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
import time

from sgvad import SGVAD

@hydra_runner(config_path=".", config_name="train_cfg")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    # 動態修改設定：要求儲存 Top 3 與 Last
    OmegaConf.set_struct(cfg, False)  # 解除唯讀限制，允許動態新增設定
    if "checkpoint_callback_params" not in cfg.exp_manager:
        cfg.exp_manager.checkpoint_callback_params = {}
        
    cfg.exp_manager.checkpoint_callback_params.save_top_k = 3
    cfg.exp_manager.checkpoint_callback_params.save_last = True
    cfg.exp_manager.checkpoint_callback_params.monitor = "val_loss"  # 監控驗證集的 Loss
    cfg.exp_manager.checkpoint_callback_params.mode = "min"          # 數值越低越好

    original_cfg = cfg.copy()
    for seed in [0]:
        cfg = original_cfg.copy()
        seed_everything(seed)
        trainer = pl.Trainer(**cfg.trainer)
        cfg.exp_manager.version=f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_seed_{seed}"
        exp_manager(trainer, cfg.get("exp_manager", None))
        full_model = EncDecClassificationModel(cfg=cfg.model, trainer=trainer)
        trainer.fit(full_model)
        if full_model.prepare_test(trainer):
            trainer.test(full_model)
        
        # 訓練結束後，自動封裝 Top 3 與 Last 模型
        exp_dir = cfg.exp_manager.get('exp_dir', 'exp_name')
        checkpoint_callback = trainer.checkpoint_callback

        # --- 處理 Top K (前三名) ---
        if hasattr(checkpoint_callback, 'best_k_models'):
            top_k_dict = checkpoint_callback.best_k_models
            logging.info(f"\n[INFO] 找到 {len(top_k_dict)} 個最佳模型，準備進行封裝...")

            # 根據 validation loss 進行由小到大排序 (確保 Top 1 確實是 Loss 最低的)
            sorted_ckpts = sorted(top_k_dict.items(), key=lambda x: x[1])

            for rank, (ckpt_path, score) in enumerate(sorted_ckpts, start=1):
                logging.info(f"-> 正在封裝 Top {rank} 模型 (val_loss: {score.item():.4f})...")
                # 載入該名次的權重
                best_model = EncDecClassificationModel.load_from_checkpoint(ckpt_path)
                sgvad_top = SGVAD(
                    preprocessor=best_model.preprocessor,
                    model=best_model.encoder,
                    cfg=cfg
                )
                out_path = f"./{exp_dir}/sgvad_top{rank}.pth"
                sgvad_top.save_ckpt(out_path)

        # --- 處理 Last (最後一輪) ---
        logging.info("\n[INFO] 正在封裝最後一輪 (Last) 模型...")
        # 訓練結束時，記憶體中的 full_model 就是最後一個 step 的狀態，直接封裝即可
        sgvad_last = SGVAD(
            preprocessor=full_model.preprocessor,
            model=full_model.encoder,
            cfg=cfg
        )
        out_path_last = f"./{exp_dir}/sgvad_last.pth"
        sgvad_last.save_ckpt(out_path_last)

        logging.info("\n[INFO] 所有模型封裝完成！")

if __name__ == '__main__':
    main()
