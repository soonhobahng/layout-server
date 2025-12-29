#!/usr/bin/env python3
"""
캐시된 모델 파일 정리 및 복사 스크립트
"""

import os
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_cached_models():
    """쿼리 파라미터가 포함된 캐시된 모델 파일들을 정리"""
    
    print("🔧 캐시된 모델 파일 정리 중...")
    
    # torch 캐시 디렉토리
    torch_cache = Path.home() / '.torch' / 'iopath_cache'
    
    if not torch_cache.exists():
        print("❌ torch 캐시 디렉토리가 없습니다.")
        return False
    
    # layoutparser 모델 디렉토리 생성
    model_dir = Path.home() / '.layoutparser' / 'models' / 'faster_rcnn_R_50_FPN_3x'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 모델 저장 경로: {model_dir}")
    
    # 캐시에서 파일 찾기
    config_found = False
    model_found = False
    
    for root, dirs, files in os.walk(torch_cache):
        for file in files:
            full_path = Path(root) / file
            
            # config 파일 찾기
            if 'config.yml' in file:
                target_config = model_dir / 'config.yml'
                if not target_config.exists():
                    shutil.copy2(full_path, target_config)
                    print(f"✅ Config 파일 복사: {full_path} -> {target_config}")
                    config_found = True
                else:
                    print(f"✓ Config 파일 이미 존재: {target_config}")
                    config_found = True
            
            # 모델 파일 찾기
            if 'model_final.pth' in file:
                target_model = model_dir / 'model_final.pth'
                if not target_model.exists():
                    print(f"📥 모델 파일 복사 중... {full_path} -> {target_model}")
                    shutil.copy2(full_path, target_model)
                    print(f"✅ 모델 파일 복사 완료")
                    model_found = True
                else:
                    print(f"✓ 모델 파일 이미 존재: {target_model}")
                    model_found = True
    
    # 결과 확인
    if config_found and model_found:
        print("\n✅ 모든 모델 파일이 준비되었습니다!")
        print(f"📍 위치: {model_dir}")
        print("   - config.yml")
        print("   - model_final.pth")
        return True
    else:
        print("\n❌ 일부 파일을 찾을 수 없습니다:")
        if not config_found:
            print("   - config.yml 누락")
        if not model_found:
            print("   - model_final.pth 누락")
        return False

def test_model_loading():
    """정리된 모델로 로딩 테스트"""
    
    print("\n🧪 모델 로딩 테스트...")
    
    model_dir = Path.home() / '.layoutparser' / 'models' / 'faster_rcnn_R_50_FPN_3x'
    config_file = model_dir / 'config.yml'
    model_file = model_dir / 'model_final.pth'
    
    if not config_file.exists() or not model_file.exists():
        print("❌ 모델 파일이 없습니다.")
        return False
    
    try:
        import layoutparser as lp
        
        # 로컬 파일로 모델 로딩
        model = lp.Detectron2LayoutModel(
            config_path=str(config_file),
            model_path=str(model_file),
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7]
        )
        
        print("✅ 모델 로딩 성공!")
        print(f"   Config: {config_file}")
        print(f"   Model: {model_file}")
        return True
        
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        return False

def main():
    print("=" * 60)
    print("Layout Parser 캐시 모델 정리 도구")
    print("=" * 60)
    
    # 1. 캐시된 파일 정리
    files_ready = fix_cached_models()
    
    # 2. 모델 로딩 테스트
    if files_ready:
        success = test_model_loading()
        
        print("\n" + "=" * 60)
        if success:
            print("🎉 완료! 이제 서버를 재시작하세요:")
            print("   uvicorn app.main:app --reload")
        else:
            print("⚠ 모델 파일은 준비되었지만 로딩에 실패했습니다.")
            print("   서버를 재시작해서 다시 시도해보세요.")
    else:
        print("\n❌ 모델 파일을 찾을 수 없습니다.")
        print("다음 명령으로 다시 다운로드하세요:")
        print("   python quick_model_fix.py")

if __name__ == "__main__":
    main()