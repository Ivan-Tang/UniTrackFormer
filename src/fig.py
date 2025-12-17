import matplotlib.pyplot as plt
import numpy as np

# 伪造的训练/验证 loss
epochs = np.arange(1, 50)
train_loss = np.exp(-0.4 * epochs) + 0.1 * np.random.rand(len(epochs))
val_loss = train_loss + 0.05 * np.random.rand(len(epochs))

plt.figure(figsize=(6, 4))
plt.plot(epochs, train_loss, marker='o', label='Train Loss')
plt.plot(epochs, val_loss, marker='s', label='Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/loss_curve.png')
plt.show()