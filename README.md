# Medical Image Segmentation with Diffusion Probabilistic Model

Đây là mã nguồn cho NCKH: ỨNG DỤNG MÔ HÌNH KHUẾCH TÁN TRONG PHÂN VÙNG ẢNH Y TẾ

## Sinh viên
| Họ và tên     | Mã sinh viên |
| ------------- | ------------ |
| [Phạm Tiến Du](https://github.com/dupham2206)  | 20020039     |
| [Phạm Gia Linh](https://github.com/phamgialinhlx) | 20020203     |
| [Trịnh Ngọc Huỳnh](https://github.com/huynhspm) | 20020054     |


## Dữ liệu

Bộ dữ liệu được sử dụng là LIDC-IDRI. LIDC-IDRI là một bộ dữ liệu y tế được sử dụng để nghiên cứu về tìm kiếm bệnh lý phổi. Nó bao gồm hàng trăm nghìn hình ảnh CT ghi lại từ các bệnh nhân mạng bệnh tật về phổi. Bộ dữ liệu này đã được sử dụng trong nhiều nghiên cứu về phát hiện và phân loại các bệnh lý phổi dựa trên hình ảnh CT, bao gồm cả các nghiên cứu sử dụng các mô hình deep learning. Bộ dữ liệu này là một tài nguyên quan trọng cho các nhà nghiên cứu trong lĩnh vực y tế và AI, vì nó cung cấp một số lượng lớn dữ liệu để hỗ trợ nghiên cứu và phát triển các mô hình AI.

## Huấn luyện 

Huấn luyện mô hình khuếch tán
```
python scripts/segmentation_train.py --data_name LUNG --data_dir input data direction --out_dir output data direction --image_size 128 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 8
```

Inference: segmentation cho ảnh đầu vào

```
python scripts/segmentation_sample.py --data_name LUNG --data_dir input data direction --out_dir output data direction --model_path saved model --image_size 128 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --num_ensemble 5 --use_ddim False
```

Trực quan hóa quá trình lấy mẫu sử dụng [Visdom](https://github.com/fossasia/visdom).
