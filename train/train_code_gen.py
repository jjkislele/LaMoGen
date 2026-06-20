import argparse
import os
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from network.lbnCodeGen import LbnGenModel
from datamodules.lbnDMT2C import LBNDataT2MModule


def load_cfg(cfg_path: str) -> dict:
    """Load YAML config file."""
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    parser = argparse.ArgumentParser(description='Unified training script for LBN code generation')
    parser.add_argument('--cfg', type=str, required=True,
                        help='Config file path, relative to the cfgs/ directory '
                             'in the project root, e.g. code_gen_hml3d.yaml')
    args = parser.parse_args()

    # Resolve cfg path: support both relative to cfgs/ and absolute path
    cfg_filename = args.cfg
    if os.path.isabs(cfg_filename):
        cfg_path = cfg_filename
    else:
        # Default: look up in cfgs/ directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg_path = os.path.join(project_root, 'cfgs', cfg_filename)

    assert os.path.exists(cfg_path), f"Config file not found: {cfg_path}"
    cfg = load_cfg(cfg_path)

    # Print the config being used
    print(f"+++ Using config: {cfg_path} +++")
    print(yaml.dump(cfg, default_flow_style=False, allow_unicode=True))

    # Seed
    seed = cfg.get('seed', 2333)
    seed_everything(seed, workers=True)

    # Extract config sections
    exp_name = cfg['exp_name']
    data_cfg = cfg['data']
    model_cfg = cfg['model']
    train_cfg = cfg['training']
    ckpt_cfg = cfg['checkpoint']
    log_cfg = cfg['logging']

    batch_size = train_cfg['batch_size']
    max_epochs = train_cfg['max_epochs']
    downsample = model_cfg.get('downsample', 4)

    # Logger
    logger = TensorBoardLogger(log_cfg['log_dir'], name=exp_name, version=0)

    # DataModule
    dm = LBNDataT2MModule(
        batch_size=batch_size,
        downsample=downsample,
        pkl_text_train=data_cfg['pkl_text_train'],
        pkl_lbn_train=data_cfg['pkl_lbn_train'],
        pkl_ids_train=data_cfg['pkl_ids_train'],
        pkl_llm_train=data_cfg['pkl_llm_train'],
        pkl_text_val=data_cfg['pkl_text_val'],
        pkl_lbn_val=data_cfg['pkl_lbn_val'],
        pkl_ids_val=data_cfg['pkl_ids_val'],
        pkl_llm_val=data_cfg['pkl_llm_val'],
        mean_path=data_cfg['mean_path'],
        std_path=data_cfg['std_path'],
        glove_path=data_cfg['glove_path'],
        num_workers=data_cfg.get('num_workers', 32),
    )

    # Model
    model = LbnGenModel(
        batch_size=batch_size,
        text_num=model_cfg.get('text_num', 1),
        code_num=model_cfg['code_num'],
        time_num=model_cfg.get('time_num', 200),
        downsample=downsample,
        debug=model_cfg.get('debug', False),
        pkeep=model_cfg.get('pkeep', 0.5),
        decay_factor=model_cfg.get('decay_factor', 0.99),
        min_sampling_prob=model_cfg.get('min_sampling_prob', 0.1),
        mask_ratio=model_cfg.get('mask_ratio', None),
        fid_ckpt_path=model_cfg['fid_ckpt_path'],
        fid_pose_dim=model_cfg.get('fid_pose_dim', 263),
        decoder_ckpt_path=model_cfg['decoder_ckpt_path'],
    )

    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        save_last=ckpt_cfg.get('save_last', True),
        every_n_train_steps=ckpt_cfg.get('every_n_train_steps', 1000),
        monitor=ckpt_cfg.get('monitor', 'val_fid'),
        filename=ckpt_cfg.get('filename', 'ckpt-{epoch:04d}-{val_fid:.2f}'),
    )

    # Trainer args
    print(f"+++ EXP: {exp_name} +++")
    extra_trainer_args = {"precision": train_cfg.get('precision', '32')}
    if torch.cuda.is_available():
        extra_trainer_args["strategy"] = "ddp_find_unused_parameters_true"
        extra_trainer_args["accumulate_grad_batches"] = train_cfg.get('accumulate_grad_batches', 4)
        print("cuda available! use all gpu in the machine")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        check_val_every_n_epoch=train_cfg.get('check_val_every_n_epoch', 1),
        log_every_n_steps=train_cfg.get('log_every_n_steps', 10),
        accelerator='gpu',
        devices=train_cfg.get('devices', [0, 1]),
        callbacks=[checkpoint_callback],
        **extra_trainer_args,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
