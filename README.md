# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement 
Image classification is a fundamental task in computer vision where an input image is assigned to one of several predefined classes.
The objective of this experiment is to build and train a Convolutional Neural Network (CNN) using a labeled image dataset and evaluate its performance using accuracy, confusion matrix, and classification report.

## Dataset

For this experiment, the CIFAR-10 dataset is used.

Total Images: 60,000

Training Images: 50,000

Test Images: 10,000

Number of Classes: 10

Image Size: 32 × 32 × 3 (RGB)

## Neural Network Model

<img width="962" height="468" alt="image" src="https://github.com/user-attachments/assets/7eb63380-a55b-4d2c-8e98-ed7ed715c7c3" />


## DESIGN STEPS

## STEP 1: Data Preparation

Import required libraries (torch, torchvision, numpy, sklearn).

Load CIFAR-10 dataset.

Normalize the images.

Create DataLoader for training and testing.

## STEP 2: Model Construction

Define CNN class inheriting from nn.Module.

Add convolution, pooling, and fully connected layers.

Define forward propagation.

## STEP 3: Model Training & Evaluation

Define Loss Function (CrossEntropyLoss).

Define Optimizer (Adam).

Train the model for required epochs.

Evaluate using Confusion Matrix and Classification Report.

Test prediction for a new image.


## PROGRAM

### Name:
### Register Number:
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)




    def forward(self, x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x

```

```python
# Initialize model, loss function, and optimizer
model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(), lr=0.001)

```

```python
## Step 3: Train the Model
def train_model(model, train_loader, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
          optimizer.zero_grad()
          outputs = model(images)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()


        print('Name:NETHRAA N')
        print('Register Number:  212224040217     ')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

## OUTPUT
### Training Loss per Epoch

<img width="334" height="201" alt="image" src="https://github.com/user-attachments/assets/5384af08-5019-44c5-a207-47c8c3468650" />


### Confusion Matrix

<img width="814" height="695" alt="image" src="https://github.com/user-attachments/assets/cf9d8b2e-647c-4876-9e11-eccfb0d021c8" />


### Classification Report

<img width="577" height="342" alt="image" src="https://github.com/user-attachments/assets/ed426ee1-9dac-4e07-b38f-2baacb97fb86" />



### New Sample Data Prediction

<img width="578" height="549" alt="image" src="https://github.com/user-attachments/assets/739facdd-d03f-4bfb-982d-27928bf525b6" />


## RESULT
The Convolutional Neural Network model was successfully developed and trained using the CIFAR-10 dataset.
