@echo off
echo ======================================================
echo StyleTTS2 Bengali - GPU Environment Setup
echo ======================================================
echo.
echo 1. Uninstalling existing torch (to avoid CPU/GPU conflicts)...
pip uninstall torch torchaudio torchvision -y
echo.
echo 2. Installing Torch with CUDA 11.8 support...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
echo.
echo 3. Installing other dependencies...
pip install -r requirements.txt
echo.
echo ======================================================
echo SETUP COMPLETE! 
echo 
echo IMPORTANT: 
echo Make sure you have installed 'espeak-ng' and added it to 
echo your Windows PATH.
echo ======================================================
pause
