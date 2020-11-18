~11월 30일 경까지 쓰던 데이터파일...  

pytorch로만 열 수 있습니다.

### 데이터 설명

이 데이터는 crop / interpolation이 되있는 데이터입니다.  
`timestamp`가 항상 index마다 `1/6ms` 만큼 떨어져있도록 interpolation 되어있고,  
linear interpolation으로 각각의 데이터들도 이동시켜져있습니다.  
참고자료: [01주차 보고서](https://docs.google.com/presentation/d/1hLxl0l5nlJBlOz03NBHoKtS03RHSklkq4zOKCgHmgeU/edit?usp=sharing)

파일 이름의 규칙은...  
- `C2`는 의미가 잘 기억 안납니다.  
- `T18`은 train 데이터에 대한 target(=label)이 18index 만큼(6index = 100ms 이므로, 300ms 떨어졌다는 뜻) 차이가 있다는 뜻.  
- `win48`은 window size가 48 index 라는 뜻.  
- `hop3`은 데이터를 만들면서 3 index 만큼씩 건너띄어(hopping)서 만들었다는 뜻.  
성능은 안좋아지겠지만, 학습 속도는 빨라질 것.

### 여는 방법

```python
import torch

torch.load('C2-T18-win48-hop1.pth`)
```

