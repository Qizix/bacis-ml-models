import numpy as np
import matplotlib.pyplot as plt

# Стани погоди
states = ["Sunny", "Rainy"]

P = np.array([[0.7, 0.3],
              [0.4, 0.6]])

current_state = 0  # 0 відповідає "Сонячно", 1 - "Дощ"

num_days = 50

state_history = [current_state]

# Симуляція марківського ланцюга
for _ in range(num_days):
    # Вибір наступного стану на основі ймовірностей переходу
    current_state = np.random.choice([0, 1], p=P[current_state])
    state_history.append(current_state)

# Візуалізація
days = range(num_days + 1)
weather_labels = [states[i] for i in state_history]

# Створюємо графік
plt.figure(figsize=(10, 6))
plt.plot(days, state_history, marker='o', linestyle='-', color='blue')

# Налаштування графіка
plt.title("")
plt.xlabel("День")
plt.ylabel(")")
plt.xticks(days) # Показуємо всі дні на осі x
plt.yticks([0, 1], states) # Замінюємо цифри на назви станів
plt.grid(True)

# Додаємо текстові мітки для кожного дня
for i, label in enumerate(weather_labels):
    plt.annotate(label, (days[i], state_history[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.show()


# Розрахунок ймовірності дощу в кожен день
rain_probabilities = []
initial_distribution = np.array([1.0, 0.0]) if state_history[0] == 0 else np.array([0.0, 1.0]) # Початковий розподіл ймовірностей
current_distribution = initial_distribution

for _ in range(num_days + 1):
    rain_probabilities.append(current_distribution[1]) # Ймовірність дощу - це другий елемент розподілу
    current_distribution = np.dot(current_distribution, P) # Перемножуємо вектор ймовірностей на матрицю переходу

# Візуалізація ймовірності дощу
plt.figure(figsize=(
    10, 6))
plt.plot(days, rain_probabilities, marker='o', linestyle='-', color='red')
plt.title("Ймовірність дощу протягом прогнозованого періоду")
plt.xlabel("День")
plt.ylabel("Ймовірність дощу")
plt.xticks(days)
plt.yticks(np.arange(0, 1.1, 0.1)) # Крок 0.1 для осі Y
plt.grid(True)
plt.show()


#Вивід на консоль результатів симуляції
print("Історія станів (0 - Сонячно, 1 - Дощ):", state_history)
print("Прогноз погоди:")
for i, state in enumerate(state_history):
    print(f"День {i}: {states[state]}")

print("\nЙмовірності дощу протягом прогнозованого періоду:")
for i, prob in enumerate(rain_probabilities):
    print(f"День {i}: {prob:.2f}")