import numpy as np
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score


# ========================================
# 1. 기본 설정
# ========================================

MODEL_CHECKPOINT = "yobi/klue-roberta-base-ynat"

NUM_LABELS = 7
MAX_LENGTH = 40

BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

TRAIN_DATA_PATH = "data/train_cleandata.csv"
TEST_DATA_PATH = "data/test_cleandata.csv"

MODEL_SAVE_PATH = "model/news_topic_model"


# ========================================
# 2. 데이터 로드
# ========================================

print("데이터를 불러오는 중...")

train = pd.read_csv(TRAIN_DATA_PATH)
test = pd.read_csv(TEST_DATA_PATH)

print(f"Train 데이터: {len(train)}개")
print(f"Test 데이터: {len(test)}개")


# ========================================
# 3. 필요한 컬럼만 추출
# ========================================

df_train = pd.DataFrame({
    "title": train["title"],
    "label": train["topic_idx"]
})

df_validation = pd.DataFrame({
    "title": test["title"],
    "label": test["topic_idx"]
})


# Pandas DataFrame → Hugging Face Dataset
dataset_train = Dataset.from_pandas(df_train)
dataset_validation = Dataset.from_pandas(df_validation)


# ========================================
# 4. Tokenizer
# ========================================

print("Tokenizer를 불러오는 중...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_CHECKPOINT
)


def preprocess_function(examples):
    return tokenizer(
        examples["title"],
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True,
    )


print("Tokenizing 중...")

encoded_dataset_train = dataset_train.map(
    preprocess_function,
    batched=True
)

encoded_dataset_validation = dataset_validation.map(
    preprocess_function,
    batched=True
)


# ========================================
# 5. Label 정의
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
# 6. 모델 생성
# ========================================

print("모델을 불러오는 중...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
)


# ========================================
# 7. 평가 함수
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
# 8. TrainingArguments
# ========================================

training_args = TrainingArguments(
    output_dir="./checkpoints",

    evaluation_strategy="epoch",
    save_strategy="epoch",

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    num_train_epochs=NUM_EPOCHS,

    weight_decay=WEIGHT_DECAY,

    load_best_model_at_end=True,
    metric_for_best_model="accuracy",

    logging_dir="./logs",
)


# ========================================
# 9. Trainer
# ========================================

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=encoded_dataset_train,
    eval_dataset=encoded_dataset_validation,

    tokenizer=tokenizer,

    compute_metrics=compute_metrics,
)


# ========================================
# 10. 모델 학습
# ========================================

print("\n===== 학습 시작 =====")

trainer.train()

print("\n===== 학습 완료 =====")


# ========================================
# 11. 최종 평가
# ========================================

print("\n===== 평가 시작 =====")

evaluation_result = trainer.evaluate()

print(evaluation_result)


# ========================================
# 12. 모델 저장
# ========================================

print("\n===== 모델 저장 =====")

trainer.save_model(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

print(
    f"모델 저장 완료: {MODEL_SAVE_PATH}"
)