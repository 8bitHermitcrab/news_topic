"""
한국어 뉴스 토픽 분류 모델 학습 및 평가

- 기본 모델: klue/roberta-base (뉴스 토픽 데이터로 학습되지 않은 일반 사전학습 모델)
- 평가 방식: train_data.csv에서 토픽 비율을 유지한 채 20%를 검증셋으로 떼어 평가
  (data/korean/test_data.csv는 대회 제출용이라 정답 라벨이 없어 평가에 쓸 수 없음)

사용 예시
  # 기본: klue/roberta-base를 학습하고 검증셋으로 평가
  python train_korean.py

  # 비교용: 학습 없이 yobi/klue-roberta-base-ynat를 같은 검증셋으로만 평가
  python train_korean.py --model yobi/klue-roberta-base-ynat --eval-only
"""

import argparse
import inspect
import json
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ========================================
# 1. 기본 설정
# ========================================

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="klue/roberta-base", help="불러올 모델 이름")
parser.add_argument("--eval-only", action="store_true", help="학습 없이 검증셋 평가만 실행")
parser.add_argument("--epochs", type=int, default=3)
args = parser.parse_args()

MODEL_CHECKPOINT = args.model

NUM_LABELS = 7
MAX_LENGTH = 40

BATCH_SIZE = 32
NUM_EPOCHS = args.epochs
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

VALIDATION_SIZE = 0.2
SEED = 42

TRAIN_DATA_PATH = "data/korean/train_data.csv"

# 서비스가 쓰는 기존 모델(model/korean/news_topic_model)을 덮어쓰지 않도록 다른 폴더에 저장
MODEL_SAVE_PATH = "model/korean/news_topic_model_klue"
RESULT_DIR = "results/korean"

set_seed(SEED)
os.makedirs(RESULT_DIR, exist_ok=True)


# ========================================
# 2. 데이터 로드와 검증셋 분리
# ========================================

print("데이터를 불러오는 중...")

data = pd.read_csv(TRAIN_DATA_PATH)

# 토픽 비율을 유지한 채(stratify) 학습 80%, 검증 20%로 나눔
train_df, valid_df = train_test_split(
    data,
    test_size=VALIDATION_SIZE,
    stratify=data["topic_idx"],
    random_state=SEED,
)

print(f"학습 데이터: {len(train_df)}개")
print(f"검증 데이터: {len(valid_df)}개")

# 어떤 문장이 검증에 쓰였는지 다시 확인할 수 있도록 저장
valid_df.to_csv(os.path.join(RESULT_DIR, "validation_split.csv"), index=False)

dataset_train = Dataset.from_pandas(
    pd.DataFrame({"title": train_df["title"].values, "label": train_df["topic_idx"].values})
)
dataset_validation = Dataset.from_pandas(
    pd.DataFrame({"title": valid_df["title"].values, "label": valid_df["topic_idx"].values})
)


# ========================================
# 3. Tokenizer
# ========================================

print(f"Tokenizer를 불러오는 중... ({MODEL_CHECKPOINT})")

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)


def preprocess_function(examples):
    # 길이는 배치마다 맞추므로(DataCollatorWithPadding) 여기서는 자르기만 함
    return tokenizer(examples["title"], max_length=MAX_LENGTH, truncation=True)


print("Tokenizing 중...")

encoded_dataset_train = dataset_train.map(preprocess_function, batched=True)
encoded_dataset_validation = dataset_validation.map(preprocess_function, batched=True)

# 한글이 알 수 없는 토큰([UNK])으로 얼마나 바뀌는지 확인 (값이 크면 토크나이저가 한국어에 맞지 않음)
unk_id = tokenizer.unk_token_id
if unk_id is not None:
    total = sum(len(ids) for ids in encoded_dataset_validation["input_ids"])
    unk = sum(ids.count(unk_id) for ids in encoded_dataset_validation["input_ids"])
    print(f"[UNK] 토큰 비율: {unk / total:.2%}")


# ========================================
# 4. Label 정의
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

label2id = {label: idx for idx, label in id2label.items()}


# ========================================
# 5. 모델 생성
# ========================================

print("모델을 불러오는 중...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
)


# ========================================
# 6. 평가 함수
# ========================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }


# ========================================
# 7. TrainingArguments
# ========================================

# transformers 버전에 따라 인자 이름이 다름 (evaluation_strategy → eval_strategy)
training_kwargs = dict(
    output_dir="./checkpoints/korean",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    report_to="none",
    seed=SEED,
)
if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
    training_kwargs["eval_strategy"] = "epoch"
else:
    training_kwargs["evaluation_strategy"] = "epoch"

training_args = TrainingArguments(**training_kwargs)


# ========================================
# 8. Trainer
# ========================================

trainer_kwargs = dict(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset_train,
    eval_dataset=encoded_dataset_validation,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)
# transformers 버전에 따라 tokenizer 인자 이름이 다름 (tokenizer → processing_class)
if "processing_class" in inspect.signature(Trainer.__init__).parameters:
    trainer_kwargs["processing_class"] = tokenizer
else:
    trainer_kwargs["tokenizer"] = tokenizer

trainer = Trainer(**trainer_kwargs)


# ========================================
# 9. 모델 학습
# ========================================

if args.eval_only:
    print("\n===== 학습 없이 평가만 실행 =====")
else:
    print("\n===== 학습 시작 =====")
    trainer.train()
    print("\n===== 학습 완료 =====")


# ========================================
# 10. 검증셋 평가
# ========================================

print("\n===== 평가 시작 =====")

prediction = trainer.predict(encoded_dataset_validation)
y_pred = np.argmax(prediction.predictions, axis=1)
y_true = prediction.label_ids

accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")
target_names = [id2label[i] for i in range(NUM_LABELS)]
report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
matrix = confusion_matrix(y_true, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(report)
print("Confusion matrix (행: 정답, 열: 예측)")
print(pd.DataFrame(matrix, index=target_names, columns=target_names))

tag = MODEL_CHECKPOINT.replace("/", "_") + ("_eval_only" if args.eval_only else "")
with open(os.path.join(RESULT_DIR, f"eval_{tag}.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "model": MODEL_CHECKPOINT,
            "eval_only": args.eval_only,
            "epochs": 0 if args.eval_only else NUM_EPOCHS,
            "train_size": len(train_df),
            "validation_size": len(valid_df),
            "accuracy": round(float(accuracy), 4),
            "macro_f1": round(float(macro_f1), 4),
            "confusion_matrix": matrix.tolist(),
            "labels": target_names,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )
with open(os.path.join(RESULT_DIR, f"report_{tag}.txt"), "w", encoding="utf-8") as f:
    f.write(report)

print(f"평가 결과 저장: {RESULT_DIR}")


# ========================================
# 11. 모델 저장 (학습한 경우에만)
# ========================================

if not args.eval_only:
    print("\n===== 모델 저장 =====")
    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"모델 저장 완료: {MODEL_SAVE_PATH}")
