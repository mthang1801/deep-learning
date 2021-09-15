import zipfile
def unzip_file(filepath) : 
  zipref = zipfile.ZipFile(filepath)
  zipref.extractall()
  zipref.close()
  print("Unzipped file")

import os
def walk_through_directory(dirpath) : 
  for dirpath, dir_names, file_names in os.walk(dirpath) : 
    print(f"Có {len(dir_names)} thư mục và {len(file_names)} tập tin trong thư mục {dirpath}")

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
def plot_loss_curves(model_history) : 
  history = model_history.history 
  acc, val_acc = history["accuracy"], history["val_accuracy"]
  loss, val_loss = history["loss"], history["val_loss"]
  plt.figure(figsize=(16,6))
  plt.subplot(121)
  plt.plot(acc,label="train accuracy")
  plt.plot(val_acc, label="val accuracy" )
  plt.title("Accuracy")
  plt.legend()

  plt.subplot(122)
  plt.plot(loss, label="trian loss")
  plt.plot(val_loss, label="val loss")
  plt.title("Loss")
  plt.legend()

import tensorflow as tf
import datetime
def create_tensorboard_callback(dir_name, experiment_name) : 
  log_dir = os.path.join(dir_name, experiment_name, datetime.datetime.now().strftime("%d%m%Y-%H%M%S"))
  tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir)
  print(f"Đã lưu tensorboard vào {log_dir}")
  return tensorboard_cb

import matplotlib.pyplot as plt
def compare_history(original_model_history, new_model_history) : 
  og_history = original_model_history.history
  new_history = new_model_history.history 

  og_acc, og_val_acc = og_history["accuracy"], og_history["val_accuracy"]
  og_loss, og_val_loss = og_history["loss"], og_history["val_loss"]

  new_acc, new_val_acc = new_history["accuracy"], new_history["val_accuracy"]
  new_loss, new_val_loss=  new_history["loss"], new_history["val_loss"]

  total_acc, total_val_acc = og_acc + new_acc , og_val_acc + new_val_acc
  total_loss, total_val_loss = og_loss + new_loss, og_val_loss + new_val_loss 

  plt.figure(figsize=(20,5))
  plt.subplot(121)
  plt.plot(total_acc, label="train accuracy")
  plt.plot(total_val_acc, label="val accuracy")
  plt.plot([original_model_history.epoch[-1], original_model_history.epoch[-1]], plt.ylim(), label="Bắt đầu tinh chỉnh")
  plt.title("Độ chính xác của mô hình")
  plt.legend()

  plt.subplot(122)
  plt.plot(total_loss, label="train loss")
  plt.plot(total_val_loss, label="val loss")
  plt.plot([original_model_history.epoch[-1], original_model_history.epoch[-1]], plt.ylim(), label="Bắt đầu tinh chỉnh")
  plt.title("Giá trị sai lêch của mô hình")
  plt.legend()
  

import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(y_true, y_preds, class_names, norm=True, savefig=False) : 
  cm = confusion_matrix(y_true, y_preds)
  cm_norm = cm.astype("float") / cm.sum(axis=1)
  
  if class_names : 
    labels = class_names 
  else : 
    labels = range(cm.shape[0])
  
  n_rows=cm.shape[0]
  n_cols=cm.shape[1]
  fig, ax = plt.subplots(figsize=(n_cols, n_rows))
  cax = ax.matshow(cm, cmap=plt.cm.Blues)  
  fig.colorbar(cax)
  ax.set(
      title="Confusion matrix", 
      xlabel="Label dự đoán", 
      ylabel="Label thực",
      xticks=range(cm.shape[0]),
      yticks=range(cm.shape[0]),
      xticklabels=labels,
      yticklabels=labels
  )
  ax.xaxis.set_ticks_position("bottom")
  ax.title.set_size(26)
  plt.xticks(rotation=70,fontsize=14)
  plt.yticks(fontsize=14)

  threshold= (cm.min() + cm.max()) / 2

  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) : 
    if norm : 
      plt.text(j,i,
            f"{cm[i,j]} ({cm_norm[i,j]*100:.2f}%)", 
            horizontalalignment="center",
            color="white" if threshold < cm[i,j] else "black"
            )
    else : 
      plt.text(j,i,
            f"{cm[i,j]}", 
            horizontalalignment="center",
            color="white" if threshold < cm[i,j] else "black"
            )
  if savefig : 
    fig.savefig("confusion_matrix.png")

