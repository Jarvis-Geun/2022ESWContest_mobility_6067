### Reference
- [Arducam IMX519](https://www.arducam.com/docs/camera-for-jetson-nano/16mp-autofocus-camera-on-jetson-nano-and-xavier-nx/)
- [IMX519 Github](https://github.com/ArduCAM/Jetson_IMX519_Focus_Example)

<br>

## Cautions!
> 동일한 젯슨 나노로 서보 모터 드라이버 사용을 위해, I2C에 관련된 몇가지 패키지를 설치하고 권한을 설정해주었다. 그 과정에서 CSI 카메라를 동작시키는 파일과 충돌이 일어나 카메라가 작동하지 않게 되었다. 정확하지는 않지만 추측하건대, CSI 카메라도 I2C 통신을 필요로 하며 되도록이면 원활한 카메라 사용을 위해 다른 패키지는 설치하지 않는 것을 권한다. 설치할 경우, 다시 초기화하여 처음부터 시작하는 것이 빠를 수 있다.


## How to run
```shell
python3 main.py -i 7 [for I2C communication, 7 or 8] -p [path of directory] -n [name of video and ppg] -t [seconds of saving video]
```
```shell
# example
# fist experiment
python3 main.py -p jarvisgeun -n jarvisgeun1 -t 780

# second experiment
python3 main.py -p jarvisgeun -n jarvisgeun2 -t 780

# third experiment
python3 main.py -p jarvisgeun -n jarvisgeun3 -t 780

...

```
