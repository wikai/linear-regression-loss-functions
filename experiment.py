from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import QuantileRegressor
from sklearn.linear_model import HuberRegressor


x = np.linspace(0, 10, 100)
y = x * 2 + 3
X = x.reshape(-1, 1)

indices5 = [10, 30, 50, 70, 99]
indices10 = [5, 10, 20, 30, 40, 50, 60, 70, 80, 99]
indices20 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
             55, 60, 65, 70, 75, 80, 85, 90, 95, 99]

percents = ["0%", "1%", "5%", "10%", "20%"]
names = ["MSE", "MAE", "Huber"]

errors = {percent: {name: [] for name in names} for percent in percents}

all_results = {}


for seed in range(42, 62):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, 100)
    y_noisy = y + noise

    model_mse = LinearRegression()
    model_mae = QuantileRegressor(quantile=0.5, alpha=0)
    model_huber = HuberRegressor(alpha=0)

    # Результаты ОДНОГО запуска.
    results = {
        "0%": {},
        "1%": {},
        "5%": {},
        "10%": {},
        "20%": {}
    }

    #0%
    model_mse.fit(X, y_noisy)
    model_mae.fit(X, y_noisy)
    model_huber.fit(X, y_noisy)

    error_mse = np.mean(np.abs(model_mse.predict(X) - y))
    error_mae = np.mean(np.abs(model_mae.predict(X) - y))
    error_huber = np.mean(np.abs(model_huber.predict(X) - y))

    results["0%"]["MSE"] = {
        "a": float(model_mse.coef_[0]),
        "b": float(model_mse.intercept_),
        "error": float(error_mse)
    }
    results["0%"]["MAE"] = {
        "a": float(model_mae.coef_[0]),
        "b": float(model_mae.intercept_),
        "error": float(error_mae)
    }
    results["0%"]["Huber"] = {
        "a": float(model_huber.coef_[0]),
        "b": float(model_huber.intercept_),
        "error": float(error_huber)
    }

    #1%
    y_edge = y_noisy.copy()
    y_edge[-1] += 15

    model_mse.fit(X, y_edge)
    model_mae.fit(X, y_edge)
    model_huber.fit(X, y_edge)

    error_mse = np.mean(np.abs(model_mse.predict(X) - y))
    error_mae = np.mean(np.abs(model_mae.predict(X) - y))
    error_huber = np.mean(np.abs(model_huber.predict(X) - y))

    results["1%"]["MSE"] = {
        "a": float(model_mse.coef_[0]),
        "b": float(model_mse.intercept_),
        "error": float(error_mse)
    }
    results["1%"]["MAE"] = {
        "a": float(model_mae.coef_[0]),
        "b": float(model_mae.intercept_),
        "error": float(error_mae)
    }
    results["1%"]["Huber"] = {
        "a": float(model_huber.coef_[0]),
        "b": float(model_huber.intercept_),
        "error": float(error_huber)
    }

    #5%
    y_out5 = y_noisy.copy()
    y_out5[indices5] += 15

    model_mse.fit(X, y_out5)
    model_mae.fit(X, y_out5)
    model_huber.fit(X, y_out5)

    error_mse = np.mean(np.abs(model_mse.predict(X) - y))
    error_mae = np.mean(np.abs(model_mae.predict(X) - y))
    error_huber = np.mean(np.abs(model_huber.predict(X) - y))

    results["5%"]["MSE"] = {
        "a": float(model_mse.coef_[0]),
        "b": float(model_mse.intercept_),
        "error": float(error_mse)
    }
    results["5%"]["MAE"] = {
        "a": float(model_mae.coef_[0]),
        "b": float(model_mae.intercept_),
        "error": float(error_mae)
    }
    results["5%"]["Huber"] = {
        "a": float(model_huber.coef_[0]),
        "b": float(model_huber.intercept_),
        "error": float(error_huber)
    }

    #10%
    y_out10 = y_noisy.copy()
    y_out10[indices10] += 15

    model_mse.fit(X, y_out10)
    model_mae.fit(X, y_out10)
    model_huber.fit(X, y_out10)

    error_mse = np.mean(np.abs(model_mse.predict(X) - y))
    error_mae = np.mean(np.abs(model_mae.predict(X) - y))
    error_huber = np.mean(np.abs(model_huber.predict(X) - y))

    results["10%"]["MSE"] = {
        "a": float(model_mse.coef_[0]),
        "b": float(model_mse.intercept_),
        "error": float(error_mse)
    }
    results["10%"]["MAE"] = {
        "a": float(model_mae.coef_[0]),
        "b": float(model_mae.intercept_),
        "error": float(error_mae)
    }
    results["10%"]["Huber"] = {
        "a": float(model_huber.coef_[0]),
        "b": float(model_huber.intercept_),
        "error": float(error_huber)
    }

    #20%
    y_out20 = y_noisy.copy()
    y_out20[indices20] += 15

    model_mse.fit(X, y_out20)
    model_mae.fit(X, y_out20)
    model_huber.fit(X, y_out20)

    error_mse = np.mean(np.abs(model_mse.predict(X) - y))
    error_mae = np.mean(np.abs(model_mae.predict(X) - y))
    error_huber = np.mean(np.abs(model_huber.predict(X) - y))

    results["20%"]["MSE"] = {
        "a": float(model_mse.coef_[0]),
        "b": float(model_mse.intercept_),
        "error": float(error_mse)
    }
    results["20%"]["MAE"] = {
        "a": float(model_mae.coef_[0]),
        "b": float(model_mae.intercept_),
        "error": float(error_mae)
    }
    results["20%"]["Huber"] = {
        "a": float(model_huber.coef_[0]),
        "b": float(model_huber.intercept_),
        "error": float(error_huber)
    }

    all_results[seed] = results
    for percent in percents:
        for name in names:
            errors[percent][name].append(results[percent][name]["error"])


#Средние ошибки и стандартные отклонения.
results_mean = {}
for percent in percents:
    results_mean[percent] = {}
    for name in names:
        values = errors[percent][name]
        results_mean[percent][name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1))
        }

print("\nПЕРВЫЙ ЗАПУСК (seed=42):")
for percent in percents:
    print(f"\nВыбросы: {percent}")
    for name in names:
        data = all_results[42][percent][name]
        print(f"  {name:5} | a={data['a']:.4f} | b={data['b']:.4f} | "
              f"error={data['error']:.4f}")

print("\nСРЕДНИЕ РЕЗУЛЬТАТЫ ПО 20 ЗАПУСКАМ:")
print(f"{'Выбросы':<10} {'MSE':>18} {'MAE':>18} {'Huber':>18}")
for percent in percents:
    cells = []
    for name in names:
        item = results_mean[percent][name]
        cells.append(f"{item['mean']:.4f} ± {item['std']:.4f}")
    print(f"{percent:<10} {cells[0]:>18} {cells[1]:>18} {cells[2]:>18}")


#CSV и строим графики.
output_dir = Path(__file__).resolve().parent / "Оптотех_2026_результаты_вариант_2"
output_dir.mkdir(exist_ok=True)

with (output_dir / "все_запуски.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow(["seed", "доля_выбросов", "модель", "a", "b", "E"])
    for seed, seed_results in all_results.items():
        for percent in percents:
            for name in names:
                item = seed_results[percent][name]
                writer.writerow([seed, percent, name, item["a"], item["b"], item["error"]])

with (output_dir / "средние_ошибки.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow(["доля_выбросов", "модель", "средняя_E", "стд_E"])
    for percent in percents:
        for name in names:
            item = results_mean[percent][name]
            writer.writerow([percent, name, item["mean"], item["std"]])

plt.rcParams["font.family"] = "DejaVu Sans"
colors = {"MSE": "#CB4539", "MAE": "#1C7A6D", "Huber": "#4169B1"}

#Пример с 10 % выбросов и seed=42.
y_example = y + np.random.default_rng(42).normal(0, 1, len(x))
y_example[indices10] += 15

fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.scatter(x, y_example, s=18, alpha=0.65, color="gray", label="Обучающие наблюдения")
ax.plot(x, y, color="black", linewidth=2, label="Исходная зависимость")
for name in names:
    item = all_results[42]["10%"][name]
    ax.plot(x, item["a"] * x + item["b"], linewidth=2,
            color=colors[name], label=name)
ax.set(xlabel="x", ylabel="y", title="10 % выбросов (+15), seed = 42")
ax.grid(alpha=0.25)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(output_dir / "рисунок_1_сравнение_моделей.png", dpi=200)
plt.close(fig)

#Средние ошибки по 20 повторам.
percent_values = [int(percent[:-1]) for percent in percents]
fig, ax = plt.subplots(figsize=(8.5, 4.6))
for name in names:
    avg = [results_mean[percent][name]["mean"] for percent in percents]
    std = [results_mean[percent][name]["std"] for percent in percents]
    ax.errorbar(percent_values, avg, yerr=std, marker="o", capsize=3,
                color=colors[name], linewidth=2, label=name)
ax.set(xlabel="Доля выбросов, %", ylabel="Средняя ошибка E",
       title="Устойчивость линейной регрессии (20 наборов шума)")
ax.set_xticks(percent_values)
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(output_dir / "рисунок_2_средние_ошибки.png", dpi=200)
plt.close(fig)

print(f"\nCSV и графики сохранены в: {output_dir}")