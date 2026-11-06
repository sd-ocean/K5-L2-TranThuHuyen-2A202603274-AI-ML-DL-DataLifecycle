# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 7/11/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Python: 3.13.18 /
PyTorch: 2.14.1+cpu /
Ultralytics: 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**File:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`. Dùng ảnh `sample_id = "traffic"`, lấy dòng `rank = 1`.

1. Record hạng 1: `class_id = 468`; `class_name = "cab"`; `rank = 1`; `score = 0.510915`; `taxonomy_name = "ImageNet-1K"`
2. Record này nói về **toàn bộ bức ảnh**, không phải một vật thể riêng biệt trong ảnh.
- Lý do là trường `task` có giá trị `image_classification`, cho biết mô hình YOLO11n-cls đang thực hiện bài toán phân loại ảnh (Image Classification). Trong bài toán này, mô hình phân tích toàn bộ bức ảnh và đưa ra các lớp dự đoán có khả năng phù hợp nhất, kèm theo điểm tin cậy (`score`) và thứ hạng (`rank`) tương ứng.
- Cụ thể, với ảnh có `sample_id = "traffic"`, mô hình dự đoán lớp `cab` ở hạng 1, với điểm tin cậy là `0.510915`. Các record ở hạng tiếp theo, chẳng hạn `minibus` ở hạng 2, cũng là những lớp dự đoán khác nhau cho cùng bức ảnh.
- Vì vậy, mỗi record biểu thị một **kết quả dự đoán lớp** của **toàn bộ bức ảnh**, chứ không đại diện cho một vật thể riêng biệt.
3. Danh sách tên lớp không phải do mô hình tự nghĩ ra trong quá trình dự đoán, mà được xác định trước và gắn với bộ dữ liệu huấn luyện (taxonomy) cũng như cấu hình của checkpoint.
- Taxonomy xác định hệ thống lớp và tên lớp.
- Checkpoint/model metadata cung cấp thông tin ánh xạ đầu ra sang các lớp tương ứng.
- Cụ thể, `taxonomy_name = "ImageNet-1K"` cho biết mô hình sử dụng bộ phân loại gồm 1.000 lớp được xác định trước. Khi dự đoán, mô hình tính điểm số cho từng lớp và ánh xạ `class_id` sang `class_name` tương ứng. Ví dụ: `class_id = 468` được ánh xạ thành `class_name = "cab"`.
- Kết luận: Mô hình chỉ dự đoán lớp phù hợp nhất từ danh sách có sẵn, không tự tạo ra tên lớp mới.
4. Cần ghi cả `class_id`, `class_name` và `taxonomy_name` vì mỗi trường cung cấp một loại thông tin khác nhau:
- `class_id`: Mã định danh của lớp, giúp máy tính xử lý và tham chiếu chính xác.
- `class_name`: Tên lớp giúp con người dễ đọc và hiểu kết quả dự đoán.
- `taxonomy_name`: Cho biết hệ thống phân loại được sử dụng, giúp xác định ý nghĩa của `class_id` và `class_name` trong bộ nhãn tương ứng.
- Kết luận: Ghi cả ba trường giúp kết quả dự đoán rõ ràng, dễ đối chiếu, thuận tiện cho xử lý dữ liệu và tránh nhầm lẫn khi sử dụng các hệ thống phân loại khác nhau.
5. Guideline cần quy định rõ tiêu chí chọn lớp đại diện cho toàn bộ bức ảnh, chẳng hạn ưu tiên chủ thể chính, vật thể nổi bật nhất hoặc nội dung chiếm phần lớn bức ảnh. Đồng thời, cần có quy tắc xử lý khi ảnh có nhiều chủ thể quan trọng hoặc không xác định được chủ thể chính để đảm bảo tính nhất quán giữa những người gán nhãn.
6. Vì score chỉ thể hiện mức độ tự tin của mô hình đối với một dự đoán, không đảm bảo dự đoán đó đúng với thực tế. Ground truth phải được xác định dựa trên dữ liệu thực tế và guideline gán nhãn, không phải kết quả dự đoán của model.
- Kết luận: Model prediction chỉ là dự đoán tham khảo, còn ground truth là nhãn chuẩn được xác định độc lập để đánh giá độ chính xác của mô hình.

## 2. Phát hiện vật thể

**File:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Dùng ảnh `sample_id = "kitchen"`. Chọn record có `class_name = "person"`, vì đây là đối tượng dễ xác định trên ảnh.  

1. Record: `class_name = "person"`; `score = 0.912625`; `bbox_xyxy = [385.33, 69.24, 498.92, 348.92]`; `bbox_width = 113.58`; `bbox_height = 279.68`
2. Bounding box bao quanh người đứng ở khu vực giữa bên phải bức ảnh.
- Góc trên-trái: tọa độ (385.33, 69.24).
- Góc dưới-phải: tọa độ (498.92, 348.92).
- Chiều rộng: 113.58 pixel.
- Chiều cao: 279.68 pixel.
- Tọa độ được tính từ góc trên-trái của ảnh, với trục X hướng sang phải và trục Y hướng xuống dưới.
3. So sánh hai ngưỡng 0.20 và 0.60: 
- threshold=0.20: 17 vật thể → ['person', 'bowl', 'bowl', 'oven', 'oven', 'person', 'bowl', 'bowl', 'cup', 'cup', 'bowl', 'spoon', 'potted plant', 'spoon', 'dining table', 'spoon', 'bottle']. Mô hình phát hiện 17 vật thể, bao gồm cả những đối tượng có confidence thấp như spoon, potted plant, dining table và bottle.
- threshold=0.60: 6 vật thể → ['person', 'bowl', 'bowl', 'oven', 'oven', 'person']. Chỉ còn 6 vật thể có confidence cao hơn, gồm person, bowl và oven.
- Nhận xét: Khi tăng confidence threshold từ 0.20 lên 0.60, số lượng prediction giảm từ 17 xuống 6. Ngược lại, khi hạ ngưỡng, mô hình giữ lại nhiều bounding box hơn, bao gồm cả những dự đoán có độ tin cậy thấp.
- Điều này cho thấy việc lựa chọn ngưỡng ảnh hưởng trực tiếp đến số lượng đối tượng được giữ lại và khối lượng công việc kiểm tra của reviewer.
4. Reviewer cần kiểm tra thêm các bounding box có confidence thấp để:
- Loại bỏ những hộp phát hiện sai (false positives).
- Kiểm tra và chỉnh sửa nhãn lớp nếu mô hình phân loại sai.
- Điều chỉnh bounding box nếu vị trí hoặc kích thước chưa chính xác.
- Phát hiện các hộp trùng lặp.
5. Quy tắc: Người gán nhãn cần vẽ bounding box bao quanh toàn bộ phần nhìn thấy của vật thể, sát với đường viền của vật thể nhất có thể, không bao gồm quá nhiều vùng nền hoặc các vật thể khác. Các hộp cần được vẽ nhất quán giữa các ảnh.
6. Trên ảnh `kitchen`, **người xuất hiện một phần ở mép bên trái ảnh**, có `score = 0.61`, bị che khuất/cắt mép. Người này bị cắt bởi ranh giới khung hình nên chỉ nhìn thấy một phần cơ thể.
- Chọn escalation (hỏi Lab Coach) thay vì tự đưa ra quy tắc ngay lập tức, vì cần xác nhận chính sách gán nhãn đối với những vật thể chỉ xuất hiện một phần trong ảnh: có cần gán nhãn hay không và bounding box cần được xử lý như thế nào.
- Sau khi Lab Coach xác nhận, quy định này có thể được bổ sung vào guideline để áp dụng thống nhất cho các ảnh khác.

## 3. Phân đoạn theo từng đối tượng

**File:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`. Dùng ảnh `sample_id = "kitchen"`. Chọn record có `class_name = "person"`, vì đây là đối tượng dễ xác định trên ảnh.

1. Record: `instance_id = "kitchen-001"`; `class_name = "person"`; `score = 0.899318`; số điểm polygon=`polygon_point_count = 348`; 3–5 điểm đầu của `polygon_xy = [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]`
2. Bounding box chỉ xác định vùng hình chữ nhật bao quanh vật thể, trong khi polygon mô tả đường viền và hình dạng thực tế của vật thể chi tiết hơn.
- Ví dụ, polygon có thể thể hiện được hình dáng đầu, vai, tay và chân của người, thay vì chỉ bao quanh toàn bộ người bằng một hình chữ nhật.
3. `instance_id` dùng để định danh một đối tượng cụ thể trong ảnh, giúp phân biệt các đối tượng khác nhau, kể cả khi chúng có cùng `class_name`. Ví dụ:
- `instance_id = "kitchen-001"`: định danh một đối tượng người cụ thể.
- `class_id = 0`: mã lớp person.
- Vì vậy, `instance_id` xác định từng đối tượng riêng biệt, không xác định loại đối tượng như `class_id` và cũng không dùng để theo dõi đối tượng qua nhiều khung hình như tracking ID.
4. Quy tắc: Người gán nhãn cần vẽ polygon sát theo đường viền của phần vật thể nhìn thấy trong ảnh, đảm bảo bao phủ đầy đủ vật thể nhưng không bao gồm vùng nền hoặc các vật thể khác. Với những phần bị che khuất, không tự ý suy đoán đường viền không nhìn thấy.
5. Trên ảnh `kitchen`, khu vực đầu và vai của người đứng giữa ảnh có chỗ biên mờ / hai vật chạm nhau / bị che, đây là nơi tóc và vai của người nằm gần các dụng cụ nhà bếp phía sau, khiến ranh giới giữa người và vật thể xung quanh khó xác định hoàn toàn.
- Chọn escalation (hỏi Lab Coach) vì cần xác nhận cách xử lý những vùng có ranh giới không rõ ràng, đặc biệt là phần tóc hoặc cơ thể bị vật thể khác che khuất. Không nên tự suy đoán đường biên vì có thể tạo ra mask không nhất quán giữa những người gán nhãn.
- Sau khi có hướng dẫn, có thể bổ sung quy tắc cụ thể vào guideline để áp dụng cho các ảnh tương tự.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp (`class_name`) đại diện cho toàn bộ ảnh. | Ảnh `traffic` có nhiều loại phương tiện; khó chọn một lớp đại diện. Model dự đoán cab với score 0.510915 nhưng không có nghĩa đây chắc chắn là nhãn đúng. | Xem toàn bộ ảnh, chọn lớp phù hợp theo guideline. | Kiểm tra nhãn có phản ánh đúng nội dung chính của ảnh và tuân thủ guideline không. |
| Phát hiện vật thể | Mỗi vật thể có một nhãn lớp và một bounding box (`class_name`, `bbox_xyxy`). | Ảnh `kitchen` có nhiều vật thể chồng lấn, bị che khuất hoặc cắt mép. Một số prediction có score thấp, dễ bị bỏ sót hoặc nhận diện sai. | Xác định từng vật thể, gán class và vẽ bounding box sát vật thể. | Kiểm tra class, vị trí, kích thước bounding box, vật thể bị bỏ sót và prediction sai. |
| Instance segmentation | Mỗi đối tượng có một nhãn lớp và một polygon/mask riêng (`class_name`, `polygon_xy`, `instance_id`). | Ranh giới giữa các vật thể khó xác định, đặc biệt khi hai vật chạm nhau hoặc bị che khuất. | Gán nhãn từng instance và vẽ polygon theo đường biên vật thể. | Kiểm tra mask có đúng hình dạng, mask của hai đối tượng bị dính vào nhau, các instance có bị gán nhầm, đường biên có chính xác, có bị thiếu hoặc lấn sang vật thể khác không. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu: Chỉ sử dụng dữ liệu được cung cấp cho mục đích bài lab, không tự ý chia sẻ, sao chép hoặc tải dữ liệu lên các nền tảng bên ngoài khi chưa được cho phép.
2. Khi gặp dữ liệu không đúng phạm vi: Dừng xử lý, không tự ý sử dụng hoặc chỉnh sửa dữ liệu và báo cho Lab Coach để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md`
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
