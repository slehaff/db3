# db3
on rpi3 jig:
ssh pi@dantrain.local
cd Desktop
export DISPLAY:=0
pkill python
python messreceive.py


on db3:
input folders: db3/scan/static/scan_im_folder/render(i)
workon tftest_env
python manage.py runserver 0.0.0.0:8000

