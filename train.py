import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from nemo.collections.asr.models import EncDecClassificationModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
import time


@hydra_runner(config_path=".", config_name="train_cfg")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
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
        
        # 訓練完成後儲存為您的 sgvad.pth 格式
        print("訓練完成，正在封裝模型參數...")
        # 建立 SGVAD 物件以便使用其 save_ckpt 方法
        sgvad_instance = SGVAD(
            preprocessor=full_model.preprocessor,
            model=full_model.encoder, # NeMo Classification Model 的主幹通常是 encoder
            cfg=cfg
        )
        sgvad_instance.save_ckpt()
        print(f"模型已成功儲存至: {os.path.abspath('./sgvad.pth')}")

if __name__ == '__main__':
    main()
