#!/usr/bin/env python3
"""
创建原始版本文件的脚本
根据修改记录还原每个文件的原始内容
"""

import os
import shutil

# 项目根目录
PROJECT_ROOT = "/root/autodl-tmp/TIA2V"
MOD_DIR = os.path.dirname(os.path.abspath(__file__))

def restore_tacm_train_util():
    """还原 diffusion/tacm_train_util.py"""
    file_path = os.path.join(PROJECT_ROOT, "diffusion/tacm_train_util.py")
    original_path = os.path.join(MOD_DIR, "original/diffusion/tacm_train_util.py")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 还原修改
    content = content.replace("# from tacm.download import load_vqgan", "from tacm.download import load_vqgan")
    content = content.replace("#self.init_first_stage_from_ckpt()", "self.init_first_stage_from_ckpt()")
    content = content.replace("    #def init_first_stage_from_ckpt(self):", "    def init_first_stage_from_ckpt(self):")
    content = content.replace("    #    self.first_stage_model = load_vqgan(self.vqgan_ckpt)", "        self.first_stage_model = load_vqgan(self.vqgan_ckpt)")
    content = content.replace("    #    for p in self.first_stage_model.parameters():", "        for p in self.first_stage_model.parameters():")
    content = content.replace("    #        p.requires_grad = False", "            p.requires_grad = False")
    content = content.replace("    #    self.first_stage_model.codebook._need_init = False", "        self.first_stage_model.codebook._need_init = False")
    content = content.replace("    #    self.first_stage_model.eval()", "        self.first_stage_model.eval()")
    content = content.replace("    #    self.first_stage_model.train = disabled_train", "        self.first_stage_model.train = disabled_train")
    content = content.replace("    #    self.first_stage_vocab_size = self.first_stage_model.codebook.n_codes", "        self.first_stage_vocab_size = self.first_stage_model.codebook.n_codes")
    
    os.makedirs(os.path.dirname(original_path), exist_ok=True)
    with open(original_path, 'w') as f:
        f.write(content)
    print(f"Created: {original_path}")

def restore_modules_init():
    """还原 tacm/modules/__init__.py"""
    file_path = os.path.join(PROJECT_ROOT, "tacm/modules/__init__.py")
    original_path = os.path.join(MOD_DIR, "original/tacm/modules/__init__.py")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 还原修改
    content = content.replace(
        "try:\n    from .lpips import LPIPS\nexcept ImportError:\n    # LPIPS module is optional, only needed for VQGAN training\n    LPIPS = None",
        "from .lpips import LPIPS"
    )
    content = content.replace(
        "try:\n    from .audioclip import AudioCLIP\nexcept ImportError:\n    # AudioCLIP module is optional, only needed when audio_emb_model='audioclip'\n    AudioCLIP = None",
        "from .audioclip import AudioCLIP"
    )
    
    os.makedirs(os.path.dirname(original_path), exist_ok=True)
    with open(original_path, 'w') as f:
        f.write(content)
    print(f"Created: {original_path}")

def restore_data_py():
    """还原 tacm/data.py"""
    file_path = os.path.join(PROJECT_ROOT, "tacm/data.py")
    original_path = os.path.join(MOD_DIR, "original/tacm/data.py")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 还原修改
    content = content.replace(
        "try:\n    from tacm.modules import AudioCLIP\nexcept ImportError:\n    # AudioCLIP is optional, only needed when audio_emb_model='audioclip'\n    AudioCLIP = None",
        "from tacm.modules import AudioCLIP"
    )
    
    os.makedirs(os.path.dirname(original_path), exist_ok=True)
    with open(original_path, 'w') as f:
        f.write(content)
    print(f"Created: {original_path}")

if __name__ == "__main__":
    print("Creating original files...")
    restore_tacm_train_util()
    restore_modules_init()
    restore_data_py()
    print("Done!")
