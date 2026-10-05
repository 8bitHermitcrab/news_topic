import numpy as np
import pandas as pd
import re

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ========================================
# 1. 기본 설정
# ========================================

MODEL_CHECKPOINT = "distilroberta-base"

NUM_LABELS = 7
MAX_LENGTH = 100

BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 4e-5
WEIGHT_DECAY = 0.01

DATA_PATH = "data/english/en_final_dataset_34000_politics_o.csv"

MODEL_SAVE_PATH = "model/english/news_topic_model"

CHECKPOINT_PATH = "./checkpoints/english"


# ========================================
# 2. 데이터 전처리 함수
# ========================================

def preprocessing(text):
    text = str(text)

    # 개행문자 제거
    text = re.sub(r'\\n', ' ', text)

    # 한글 제거
    text = re.sub(r'[가-힣ㄱ-ㅎㅏ-ㅣ]', ' ', text)

    # 중복 공백 제거
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# ========================================
# 3. 데이터 로드
# ========================================

print("데이터를 불러오는 중...")

data = pd.read_csv(DATA_PATH)

print(f"전체 데이터: {len(data)}개")


# ========================================
# 4. 데이터 정리
# ========================================

# topic_idx → label
data = data.rename(
    columns={
        "topic_idx": "label"
    }
)

# label 정수형 변환
data["label"] = data["label"].astype(int)

# 필요없는 컬럼 제거
data = data.drop(
    ["Unnamed: 0", "title_len"],
    axis=1
)

# 제목 전처리
data["text"] = data["title"].map(preprocessing)


# 필요한 컬럼만 사용
data = data[["text", "label"]]

print("\n데이터 확인:")
print(data.head())


# ========================================
# 5. Train / Validation 분리
# ========================================

print("\nTrain / Validation 데이터를 분리하는 중...")

train_df, eval_df = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

print(f"Train 데이터: {len(train_df)}개")
print(f"Validation 데이터: {len(eval_df)}개")


# ========================================
# 6. Pandas → Hugging Face Dataset
# ========================================

train_dataset = Dataset.from_pandas(
    train_df,
    preserve_index=False
)

eval_dataset = Dataset.from_pandas(
    eval_df,
    preserve_index=False
)


# ========================================
# 7. Label 정의
# ========================================

id2label = {
    0: "IT과학",
    1: "경제",
    2: "사회",
    3: "생활문화",
    4: "세계",
    5: "스포츠",
    6: "정치",
}

label2id = {
    label: idx
    for idx, label in id2label.items()
}


# ========================================
# 8. Tokenizer
# ========================================

print("\nTokenizer를 불러오는 중...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_CHECKPOINT
)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


print("Tokenizing 중...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

eval_dataset = eval_dataset.map(
    tokenize_function,
    batched=True
)


# ========================================
# 9. 불필요한 컬럼 제거
# ========================================

train_dataset = train_dataset.remove_columns(
    ["text"]
)

eval_dataset = eval_dataset.remove_columns(
    ["text"]
)


# ========================================
# 10. 모델 생성
# ========================================

print("\n모델을 불러오는 중...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
)


# ========================================
# 11. 평가 함수
# ========================================

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = np.argmax(
        predictions,
        axis=1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {
        "accuracy": accuracy
    }


# ========================================
# 12. TrainingArguments
# ========================================

training_args = TrainingArguments(
    output_dir=CHECKPOINT_PATH,

    eval_strategy="epoch",

    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    num_train_epochs=NUM_EPOCHS,

    weight_decay=WEIGHT_DECAY,

    logging_steps=100,

    load_best_model_at_end=False,
)


# ========================================
# 13. Trainer
# ========================================

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=eval_dataset,

    compute_metrics=compute_metrics,
)


# ========================================
# 14. 모델 학습
# ========================================

print("\n===== 학습 시작 =====")

trainer.train()

print("\n===== 학습 완료 =====")


# ========================================
# 15. 최종 평가
# ========================================

print("\n===== 평가 시작 =====")

evaluation_result = trainer.evaluate()

print(evaluation_result)


# ========================================
# 16. 모델 저장
# ========================================

print("\n===== 모델 저장 =====")

trainer.save_model(
    MODEL_SAVE_PATH
)

tokenizer.save_pretrained(
    MODEL_SAVE_PATH
)

print(
    f"모델 저장 완료: {MODEL_SAVE_PATH}"
)