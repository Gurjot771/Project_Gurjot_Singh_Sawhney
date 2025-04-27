import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from dataset import create_data_loaders  # Import the dataloaders
from model import Resnet  # Import the model
from config import epochs, learning_rate, weight_decay  # Import config variables

def train_model(model, num_epochs, train_loader, loss_fn, optimizer): 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for i, (inputs, labels) in enumerate(train_loader): 
            print(f"Epoch {epoch + 1}, Batch {i + 1}/{len(train_loader)} (Train)", end='\r')
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)  # Use 'model' instead of 'net'
            loss = loss_fn(outputs, labels)  # Use loss_fn
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        print(f'\nEpoch {epoch + 1}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%', end=' ')

        # Validation
        model.eval()  
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        val_y_true_indices = []
        val_y_pred_indices = []
        with torch.no_grad():
            for images, labels in val_loader:  # changed data to images, labels
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)  # Use 'model' instead of 'net'
                loss = loss_fn(outputs, labels)  # Use loss_fn
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                val_y_true_indices.extend(labels.cpu().numpy())
                val_y_pred_indices.extend(predicted.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        val_precision_macro = precision_score(val_y_true_indices, val_y_pred_indices, average='macro', zero_division=0)
        val_recall_macro = recall_score(val_y_true_indices, val_y_pred_indices, average='macro', zero_division=0)
        val_f1_macro = f1_score(val_y_true_indices, val_y_pred_indices, average='macro', zero_division=0)
        print(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, Val Precision (Macro): {val_precision_macro:.4f}, Val Recall (Macro): {val_recall_macro:.4f}, Val F1 (Macro): {val_f1_macro:.4f}')

        # Early Stopping Check and Best Model Saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()  # Use 'model' instead of 'net'
            trigger_times = 0
        else:
            trigger_times += 1
            print(f"Validation loss did not improve for {trigger_times} epochs.")

        if trigger_times >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            model.load_state_dict(best_model_state)  # Use 'model' instead of 'net'
            break

    print('Finished Training')

    # --- Evaluation on Test Set with Added Metrics ---
    model.eval()  # Ensure model is in eval mode
    y_true_indices = []
    y_pred_indices = []
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            y_true_indices.extend(labels.cpu().numpy())
            y_pred_indices.extend(predicted.cpu().numpy())

    test_accuracy = accuracy_score(y_true_indices, y_pred_indices)
    test_precision_macro = precision_score(y_true_indices, y_pred_indices, average='macro', zero_division=0)
    test_recall_macro = recall_score(y_true_indices, y_pred_indices, average='macro', zero_division=0)
    test_f1_macro = f1_score(y_true_indices, y_pred_indices, average='macro', zero_division=0)

    # Convert integer predictions to class names
    y_true_names = [class_names[i] for i in y_true_indices]
    y_pred_names = [class_names[i] for i in y_pred_indices]

    test_report = classification_report(y_true_names, y_pred_names, digits=4, zero_division=0)

    print(f'Accuracy of the network on the test images: {test_accuracy * 100:.2f}%')
    print(f'Macro-averaged Precision on the test images: {test_precision_macro:.4f}')
    print(f'Macro-averaged Recall on the test images: {test_recall_macro:.4f}')
    print(f'Macro-averaged F1-Score on the test images: {test_f1_macro:.4f}')
    print("\nClassification Report on the test images:\n", test_report)

    # --- Save the Best Model ---
    torch.save(model.state_dict(), '_final_weights.pth')  # Save the final weights in the root directory
    #if best_model_state is not None:
    #    torch.save(best_model_state, '_checkpoints/animal_classification_model_best_resnet18.pth')
    #else:
    #    torch.save(model.state_dict(), '_checkpoints/animal_classification_model_last_resnet18.pth')


'''if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    train_dataset, val_dataset, test_dataset,class_names=create_image_datasets(main_dir)
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset)
    model = TheModel(num_classes=len(class_names)).to(device)  # Instantiate the model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # Use config variables
    #model, num_epochs, train_loader, loss_fn, optimizer
    train_model(model, epochs, train_loader, criterion, optimizer)  # Call the training function'''