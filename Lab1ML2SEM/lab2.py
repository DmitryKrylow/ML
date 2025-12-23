import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


def main():
    clarity = np.linspace(0, 100, 101)

    a, b, c = map(int, input('Введите a,b,c: ').split())
    clear_speech = fuzz.trimf(clarity, [a, b, c])

    unclear_speech = 1 - clear_speech

    x_inp = float(input("Введите процент разборчивости речи (0–100): "))

    mu_clear = fuzz.interp_membership(clarity, clear_speech, x_inp)
    mu_unclear = 1 - mu_clear

    print("\n--- Результат ---")
    print(f"Разборчивая речь: {mu_clear:.2f}")
    print(f"Неразборчивая речь (дополнение): {mu_unclear:.2f}")

    plt.plot(clarity, clear_speech, label="Разборчивая речь")
    plt.plot(clarity, unclear_speech, label="Неразборчивая речь (дополнение)", linestyle="--")
    plt.scatter([x_inp], [mu_clear])
    plt.scatter([x_inp], [mu_unclear])

    plt.title("Дополнение нечёткого множества для чёткости речи")
    plt.xlabel("Процент разборчивости")
    plt.ylabel("Степень принадлежности")
    plt.legend()
    plt.grid(True)
    plt.show()


main()
