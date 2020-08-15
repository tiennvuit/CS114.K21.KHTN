<h1>Đồ án cuối kỳ môn học Máy học.<h1>

## Bài toán: Phát hiện khuôn mặt giả mạo trong ảnh/video

## Mô tả bài toán

- Phát biểu bài toán: Phát hiện khuôn mặt trong ảnh /video có phải khuôn mặt giả mạo hay không.

- Input: ảnh chứa khuôn mặt người được thu từ camera.

- Output: gán nhãn 0 hoặc 1 cho khuôn mặt trong ảnh, trong đó:

  + Nhãn 0 (Fake) thể hiện khuôn mặt trong ảnh là khuôn mặt được hiện thị trên màn hình điện thoại.
  
  + Nhãn 1 (Real) thể hiện khuôn mặt trong ảnh là khuôn mặt của người thật đang đứng trước camera


## Tại sao cần phát hiện khuôn mặt giả mạo ?

- Các hệ thống nhận diện khuôn mặt được áp dụng trong các ứng dụng điểm danh, thanh toán nhằm xác định danh tính của con người thông khuôn mặt của họ. 

- Hệ thống nhận dạng khuôn mặt có thể bị đánh lừa bằng cách sử dụng bức ảnh chân dung của một người đã được chụp lại bằng điện thoại di động hay ảnh được in ra. 

- Vì vậy chúng ta cần phải phân loại xem khuôn mặt được đặt trước camera có phải là khuôn mặt giả mạo hay không để đảm bảo tính an toàn của hệ thống.

## Dữ liệu cho bài toán

### Cách xây dựng dữ liệu

- Dữ liệu được thu thập gồm có hai lớp là ảnh chứa khuôn mặt giả thật (0 - Real) và khuôn mặt giả mạo (1 - Fake).

- Dữ liệu ảnh chứa khuông mặt thật được rút trích từ video selfie ghi lại khuôn mặt thật của một/nhiều đối tượng. 

- Dữ liệu khuôn mặt giả mạo được xây dựng từ video ghi lại khuôn mặt giả mạo của một/nhiều đối tượng.

  
## Mô tả tập dữ liệu
- Kích thước tập dữ liệu: 1916 ảnh.
- Số lượng dữ liệu của mỗi lớp:

  + Fake face: 1115
  + Real face: 806

- Chi tiết mô tả về quá trình xây dựng và tiền xử lý dữ liệu  xem tại file [notebook](https://nbviewer.jupyter.org/github/tiennvuit/CS114.K21.KHTN/blob/master/Capstone_FakeFaceDetection/LivenessDetection_Report.ipynb).
*Truy cập dữ liêu qua [drive](https://drive.google.com/drive/folders/1P3uO1lQrTTdc8f0cuSVmOYjmJae09Imt?usp=sharing).*


## Mô tả cách rút trích đặc trưng

Có hai hướng tiếp cận bài toán này bằng phương pháp máy học:

- Sử dụng đặc trưng do chuyên gia đề xuất (hand-crafted feature)

- Hướng tiếp cận dựa trên đặc trưng học sâu (deep learning).

Thử nghiệm bài toán này với cả hai phương pháp rút trích đặc trưng.

-----
- Video demo:https://youtu.be/FJNaJZOt7Ds
- File báo cáo (định dạng pptx):https://docs.google.com/presentation/d/1do372Q5krMrznOaKZ7KrTZuhnzoVVpHSGYOanMERr-E/edit?usp=sharing
