import numpy as np
import matplotlib.pyplot as plt

# Load training and validation losses
loss_train = np.load('metrics_M_SOTA_CIFAR10/4000/trades_attack_wrn_seed_501/4000_LOSS_epoch_train.npy')
loss_val = np.load('metrics_M_SOTA_CIFAR10/4000/trades_attack_wrn_seed_501/4000_LOSS_epoch_val.npy')

# Load training and validation accuracies
acc_train = np.load('metrics_M_SOTA_CIFAR10/4000/trades_attack_wrn_seed_501/4000_accuracy_per_epoch_train.npy')
acc_val = np.load('metrics_M_SOTA_CIFAR10/4000/trades_attack_wrn_seed_501/4000_accuracy_per_epoch_val.npy')


plt.plot(loss_train, label='Training Loss')
plt.plot(loss_val, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.savefig('loss_plot.png')  # Save plot to file
plt.close()  # Close plot to prevent it from being displayed interactively

# Plot training and validation accuracies
plt.plot(acc_train, label='Training Accuracy')
plt.plot(acc_val, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')  # Save plot to file
plt.close() 