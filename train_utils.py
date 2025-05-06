import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def train_model(model, optimizer, criterion, device, train_loader, val_loader, num_epochs=30):
    model.to(device)
    best_val_acc = 0.0
    best_model_state = None
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.unsqueeze(1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
            
        # Evaluate on train + val
        train_acc = train_accuracy_model(model, train_loader, device)
        val_acc = test_model(model, val_loader, device)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        

        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {running_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
    
    model.load_state_dict(best_model_state)
    print(f"\nBest Val Accuracy: {best_val_acc:.2f}%")
    return model, history

def test_model(model, test_loader, device):
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    # no need for gradients in testing
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.unsqueeze(1)
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # calculate outputs by running images through the network
            outputs = model(inputs)
            
            # the class with the highest value is prediction
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    return acc

def train_accuracy_model(model, train_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.unsqueeze(1)
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    return acc


def plot_history(history, model_name="Model"):
    
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, history['train_loss'], label='Training Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title(f'{model_name} Training Loss')
    plt.legend()
    plt.show()

    # Accuracy
    plt.figure()
    plt.plot(epochs, history['train_acc'], label='Training Acc')
    plt.plot(epochs, history['val_acc'],   label='Validation Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
    plt.title(f'{model_name} Train vs. Val Accuracy')
    plt.legend()
    plt.show()
    
def plot_confusion_matrix(model, data_loader, device, title):
    class_names = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']

    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.unsqueeze(1).to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    plt.figure(figsize=(8, 6))
    disp.plot(cmap='Blues', values_format='d')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()