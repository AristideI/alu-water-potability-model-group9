# Water Portability Model

## Group 9 - Assignment

**Date**: 6th October 2024  
**Group Members**:
- Guled Hassan Warsameh
- Cynthia Nekesa
- Aristide Isingizwe
- Dohessiekan Xavier Gnondoyi

## Task Allocation
| Member        | Task                                  |
|---------------|---------------------------------------|
| Xavier        | Data handler (Tasks 1-3)              |
| Gulad         | Vanilla Model Implementer (Task 4)    |
| Cynthia       | Model Optimizer (Tasks 1-5)           |
| Aristide & Gulad | Model Optimizer (Tasks 6 & 7)      |
| All members   | Error Analysis and Model Evaluation (Task 8) |

---

## Google Colab Link
[Google Colab Link](https://colab.research.google.com/drive/1CkJmHdrA2GG9EWXHI_uLuPwcS9Mj1j0p?usp=chrome_ntp#scrollTo=vXlj4Qzwgnwa)

## Data Preprocessing

### Handling Missing Values
We chose to handle missing values by filling them with the median rather than dropping rows or columns with null values. This approach was selected because dropping rows would have resulted in a loss of over 1,000 rows, which is too significant. Using the median allowed us to retain more data while minimizing the impact of missing values on the model's performance.

```python
# df = df.dropna(inplace=False)  # Dropping rows was avoided
df = df.fillna(df.median(), inplace=False)  # Filling missing values with the median
```

## Feature Scaling

### StandardScaler vs. MinMaxScaler
We chose to use **StandardScaler** over **MinMaxScaler** to standardize the dataset. The **StandardScaler** centers the data around the mean with unit variance, which is more suitable when the dataset follows a normal distribution. In contrast, **MinMaxScaler** scales the data between a range (usually 0 and 1), which can distort the data if there are significant outliers.

---

## Regularization

### L1 vs. L1_L2 Regularization
We applied **L1** and **L2** regularization separately rather than using **L1_L2**. Using **L1** regularization helps promote sparsity by driving some weights to zero, making it easier to interpret which features are important. In contrast, **L1_L2** (Elastic Net) combines both but can be more complex to tune and may not offer clear benefits in all cases, particularly when feature selection is important.

## Optimizers

### Adamax vs. Adam and RMSprop
We opted for the **Adamax** optimizer instead of **Adam** or **RMSprop**. Adamax is a variant of Adam based on the infinity norm, which performs better in certain cases where Adam might have convergence issues. Here’s a brief comparison:

- **Adam**: A widely-used optimizer that combines momentum and adaptive learning rates. Suitable for most tasks.
- **RMSprop**: Focuses on adaptive learning rates and works well with non-stationary data but may suffer from slow convergence.
- **Adamax**: Handles the infinity norm and performs better when dealing with large gradients or outliers in the data.

---

## Model Training and Callbacks

We implemented the following callbacks to enhance the model's training:

- **EarlyStopping**: Monitors `val_loss` and stops training when no improvement is seen after 20 epochs, restoring the best weights.
  
    ```python
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    ```

- **ModelCheckpoint**: Saves the best model based on validation performance.
  
    ```python
    check_point = ModelCheckpoint("training/model.{epoch:03d}.keras", save_best_only=True)
    ```

- **ReduceLROnPlateau**: Reduces the learning rate by a factor of 0.2 if there’s no improvement in `val_loss` for 5 epochs, with a minimum learning rate of 0.0001.
  
    ```python
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    ```
