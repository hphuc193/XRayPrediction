import streamlit as st
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Dữ liệu nguyên nhân và cách giảm tình trạng bệnh
disease_info = {
    "Atelectasis": {
        "cause": "Xẹp phổi thường do tắc nghẽn đường thở hoặc áp lực từ bên ngoài phổi (ví dụ: tràn khí màng phổi).",
        "remedy": "Tăng cường hô hấp sâu, sử dụng máy thở áp lực dương nếu cần, và điều trị nguyên nhân cơ bản (như loại bỏ tắc nghẽn)."
    },
    "Cardiomegaly": {
        "cause": "Tim to có thể do cao huyết áp, bệnh van tim, hoặc suy tim.",
        "remedy": "Kiểm soát huyết áp, dùng thuốc theo chỉ định bác sĩ, và duy trì lối sống lành mạnh (tập thể dục nhẹ, ăn ít muối)."
    },
    "Consolidation": {
        "cause": "Củng cố phổi thường do nhiễm trùng (viêm phổi) hoặc tích tụ dịch trong phế nang.",
        "remedy": "Dùng kháng sinh nếu do vi khuẩn, nghỉ ngơi, và đảm bảo thông thoáng đường thở."
    },
    "Edema": {
        "cause": "Phù phổi do suy tim, áp lực trong mạch máu phổi tăng cao.",
        "remedy": "Dùng thuốc lợi tiểu, kiểm soát lượng nước uống, và điều trị bệnh tim nền."
    },
    "Effusion": {
        "cause": "Tràn dịch màng phổi do nhiễm trùng, ung thư, hoặc suy tim.",
        "remedy": "Chọc dò dịch nếu cần, điều trị nguyên nhân (kháng sinh, hóa trị), và theo dõi y tế."
    },
    "Emphysema": {
        "cause": "Khí phế thũng do tổn thương phế nang, thường liên quan đến hút thuốc lá.",
        "remedy": "Ngừng hút thuốc, dùng thuốc giãn phế quản, và tập thở phục hồi chức năng phổi."
    },
    "Fibrosis": {
        "cause": "Xơ phổi do tiếp xúc lâu dài với chất độc hoặc bệnh tự miễn.",
        "remedy": "Dùng thuốc chống viêm, oxy liệu pháp, và tránh tiếp xúc với tác nhân gây xơ."
    },
    "Hernia": {
        "cause": "Thoát vị cơ hoành do yếu cơ hoặc áp lực trong ổ bụng tăng cao.",
        "remedy": "Phẫu thuật nếu nghiêm trọng, giảm áp lực bụng (tránh táo bón, nâng vật nặng)."
    },
    "Infiltration": {
        "cause": "Thâm nhiễm do nhiễm trùng, viêm, hoặc khối u trong phổi.",
        "remedy": "Xác định nguyên nhân qua xét nghiệm, điều trị bằng kháng sinh hoặc thuốc đặc hiệu."
    },
    "Mass": {
        "cause": "Khối u trong phổi có thể do ung thư hoặc u lành tính.",
        "remedy": "Sinh thiết để xác định, phẫu thuật hoặc hóa trị nếu là ung thư."
    },
    "Nodule": {
        "cause": "Nốt phổi có thể do nhiễm trùng cũ hoặc dấu hiệu sớm của ung thư.",
        "remedy": "Theo dõi định kỳ bằng CT, sinh thiết nếu nghi ngờ ác tính."
    },
    "Pleural_Thickening": {
        "cause": "Dày màng phổi do viêm mãn tính, nhiễm trùng, hoặc tiếp xúc amiăng.",
        "remedy": "Điều trị viêm nếu có, theo dõi để phát hiện biến chứng."
    },
    "Pneumonia": {
        "cause": "Viêm phổi do vi khuẩn, virus hoặc nấm.",
        "remedy": "Dùng kháng sinh/kháng virus tùy nguyên nhân, nghỉ ngơi, và uống đủ nước."
    },
    "Pneumothorax": {
        "cause": "Tràn khí màng phổi do chấn thương hoặc phổi bị thủng tự nhiên.",
        "remedy": "Dẫn lưu khí nếu nghiêm trọng, nghỉ ngơi, và tránh áp lực lên phổi."
    },
    "No Finding": {
        "cause": "Không phát hiện bất thường trên X-quang.",
        "remedy": "Duy trì sức khỏe phổi bằng cách tránh khói bụi và tập thể dục đều đặn"
    }
}

# Load model
def load_model(model_path, num_labels):
    model = models.densenet121(pretrained=False)
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Không định nghĩa classifier ngay, load state_dict trước
    state_dict = torch.load(model_path, map_location="cpu")
    
    # Kiểm tra số nhãn trong state_dict
    original_num_labels = state_dict["classifier.1.weight"].shape[0]
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1024, original_num_labels)  # Giữ nguyên số nhãn gốc
    )
    model.load_state_dict(state_dict)
    
    # Thêm lớp ánh xạ từ original_num_labels về num_labels (15)
    mapping_layer = nn.Linear(original_num_labels, num_labels)
    full_model = nn.Sequential(model, mapping_layer)
    full_model.eval()
    return full_model

# Dự đoán
def predict(image, model, all_labels):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    img = Image.open(image).convert("L")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        preds = torch.sigmoid(output).numpy()[0]
    # In toàn bộ dự đoán để kiểm tra
    print("Raw predictions:", {all_labels[i]: round(preds[i], 4) for i in range(len(all_labels))})
    # Giảm ngưỡng xuống 0.2 để tăng độ nhạy (điều chỉnh từ 0.9 về 0.2 như file gốc)
    results = {all_labels[i]: round(preds[i], 2) for i in range(len(all_labels)) if preds[i] > 0.8}
    return results, preds  # Trả về cả results (dự đoán vượt ngưỡng) và preds (toàn bộ xác suất)

# Lưu dữ liệu với kiểm tra trùng lặp
def save_data(image, prediction, user_label=None):
    os.makedirs("C:/XRayProject/retrain_data", exist_ok=True)
    img_path = f"C:/XRayProject/retrain_data/{image.name}"
    pil_image = Image.open(image)
    pil_image.save(img_path)
    
    csv_path = "C:/XRayProject/retrain_data/data.csv"
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        if img_path in df_existing["image"].values:
            print(f"Image {img_path} already exists in data.csv, skipping...")
            return
    
    data = {"image": img_path, "prediction": str(prediction), "user_label": str(user_label) if user_label else None}
    df = pd.DataFrame([data])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

# Giao diện chính
st.title("Phát hiện bệnh từ ảnh X-quang ngực")

all_labels = sorted([
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "No Finding", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"
])

menu = ["Dự đoán bệnh", "Thông số model", "Huấn luyện thêm"]
choice = st.sidebar.selectbox("Chọn chức năng", menu)

model_dir = "C:/XRayProject/models/"
model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")] if os.path.exists(model_dir) else []
if not model_files:
    st.error("Không tìm thấy model nào trong thư mục C:/XRayProject/models/")
else:
    selected_model = st.selectbox("Chọn Phiên Bản", model_files)
    model_path = os.path.join(model_dir, selected_model)
    model = load_model(model_path, len(all_labels))
    print("Model loaded successfully from:", model_path)

if choice == "Dự đoán bệnh":
    uploaded_file = st.file_uploader("Tải lên ảnh X-quang", type=["png", "jpg", "jpeg"])
    if uploaded_file and model_files:
        st.image(uploaded_file, caption="Ảnh đã tải lên", use_column_width=True)
        prediction, raw_preds = predict(uploaded_file, model, all_labels)  # Lấy cả results và raw_preds
        st.write("**Kết quả dự đoán:**")
        if prediction:
            for disease, prob in prediction.items():
                st.write(f"- {disease}: Xác suất {prob}")
                st.write(f"  *Nguyên nhân*: {disease_info[disease]['cause']}")
                st.write(f"  *Cách giảm tình trạng*: {disease_info[disease]['remedy']}")
        else:
            st.write("Không phát hiện bệnh nào (xác suất < 0.2)")
            st.write(f"  *Nguyên nhân*: {disease_info['No Finding']['cause']}")
            st.write(f"  *Cách giảm tình trạng*: {disease_info['No Finding']['remedy']}")

        # Vẽ biểu đồ cột hiển thị xác suất của tất cả các nhãn
        st.subheader("Biểu đồ xác suất các bệnh")
        fig, ax = plt.subplots(figsize=(10, 6))
        probabilities = raw_preds * 100  # Chuyển đổi sang phần trăm
        ax.bar(all_labels, probabilities, color='skyblue')
        ax.set_xlabel("Bệnh")
        ax.set_ylabel("Xác suất (%)")
        ax.set_title("Xác suất dự đoán các bệnh")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

        if st.checkbox("Dự đoán sai?"):
            user_labels = st.multiselect("Chọn nhãn đúng", all_labels)
            if st.button("Lưu nhãn"):
                save_data(uploaded_file, prediction, user_labels)
                st.success("Đã lưu nhãn để huấn luyện lại!")
        else:
            save_data(uploaded_file, prediction)

elif choice == "Thông số model":
    if not model_files:
        st.error("Chưa có model nào để hiển thị!")
    else:
        st.subheader("Thông tin model")
        acc_file = os.path.join(model_dir, selected_model.replace(".pth", "_acc.txt"))
        if os.path.exists(acc_file):
            with open(acc_file, "r") as f:
                accuracy = float(f.read())
            st.write(f"**Độ chính xác:** {accuracy:.2%}")
        
        plot_dir = "C:/XRayProject/plots/"
        if os.path.exists(plot_dir):
            loss_plot = os.path.join(plot_dir, "loss_plot.png")
            accuracy_plot = os.path.join(plot_dir, "accuracy_plot.png")
            if os.path.exists(loss_plot):
                st.image(loss_plot, caption="Biểu đồ Loss", use_column_width=True)
            if os.path.exists(accuracy_plot):
                st.image(accuracy_plot, caption="Biểu đồ Accuracy", use_column_width=True)

elif choice == "Huấn luyện thêm":
    st.subheader("Huấn luyện thêm với dữ liệu mới")
    if os.path.exists("C:/XRayProject/retrain_data/data.csv"):
        df = pd.read_csv("C:/XRayProject/retrain_data/data.csv")
        st.write(f"Tìm thấy {len(df)} mẫu dữ liệu mới")
        if st.button("Bắt đầu huấn luyện"):
            from train_model import retrain_model
            new_model_path = retrain_model(model, model_path, all_labels)
            st.success(f"Đã huấn luyện xong, model mới lưu tại: {new_model_path}")
    else:
        st.warning("Chưa có dữ liệu mới để huấn luyện!")