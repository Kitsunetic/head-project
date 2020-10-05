"""
이미 crop, interpolation된 csv파일을 x/y로 나눠서 hdf5 파일 하나로 합친다.
hopping을 너무 짧게 하면 기존 데이터를 외워서 과적합이 발생하지 않을지?
파일 이름을 input3-hop6-,....hdf5 이런식으로 짓자
"""
