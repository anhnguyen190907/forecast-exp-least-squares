import math
import numpy as np
import matplotlib.pyplot as plt

# ---------- Bước 1: Dữ liệu và chia train/test ----------
months = np.arange(1, 13, dtype=float)
sales = np.array([339.7, 440.4, 315.9, 439.3, 401.3, 437.4,
                  575.5, 407.6, 682.0, 475.3, 581.3, 646.9])

x_train, y_train = months[:9], sales[:9]    # tháng 1-9
x_test, y_test = months[9:], sales[9:]      # tháng 10-12


# ---------- Bước 2: Các hàm cần thiết ----------
def fit_exponential(x, y):
    """Xấp xỉ y = a*exp(b*x) bằng bình phương tối thiểu sau khi lấy ln."""
    z = np.log(y)                           # z = ln y
    n = len(x)
    # Hệ phương trình chuẩn: M @ [ln a, b] = rhs
    M = np.array([[n,       x.sum()],
                  [x.sum(), (x ** 2).sum()]])
    rhs = np.array([z.sum(), (x * z).sum()])
    ln_a, b = np.linalg.solve(M, rhs)
    return math.exp(ln_a), b, ln_a


def predict_exp(x, a, b):
    return a * np.exp(b * x)


def mae(y, y_hat):
    return np.mean(np.abs(y - y_hat))


def rmse(y, y_hat):
    return math.sqrt(np.mean((y - y_hat) ** 2))


def mape(y, y_hat):
    return np.mean(np.abs((y - y_hat) / y)) * 100


def evaluate(y, y_hat):
    return mae(y, y_hat), rmse(y, y_hat), mape(y, y_hat)


# ---------- Bước 3: Chạy, in kết quả, vẽ hình ----------
a, b, ln_a = fit_exponential(x_train, y_train)
print("=== Bình phương tối thiểu hàm mũ ===")
print(f"ln a = {ln_a:.4f}, a = {a:.4f}, b = {b:.4f}")
print(f"Tăng trưởng mỗi tháng: {(math.exp(b) - 1) * 100:.4f}%")

print(f"\n{'Tháng':>6} | {'Thực tế':>10} | {'Dự báo':>10} | {'Sai số tuyệt đối':>16}")
print("-" * 52)
for x, y in zip(months, sales):
    y_hat = predict_exp(x, a, b)
    tag = "train" if x <= 9 else "test"
    print(f"{int(x):>6} | {y:>10.4f} | {y_hat:>10.4f} | {abs(y - y_hat):>16.4f}  ({tag})")

# Sai số của mô hình hàm mũ
res = {"Hàm mũ": (evaluate(y_train, predict_exp(x_train, a, b)),
                  evaluate(y_test, predict_exp(x_test, a, b)))}

# Đa thức bậc 1, 2, 3 để so sánh (khớp với thành viên làm mục 3.3)
polys = {}
for d in (1, 2, 3):
    p = np.poly1d(np.polyfit(x_train, y_train, d))
    polys[d] = p
    res[f"Đa thức bậc {d}"] = (evaluate(y_train, p(x_train)),
                               evaluate(y_test, p(x_test)))

print(f"\n{'Mô hình':<16} | {'Tập':<5} | {'MAE':>10} | {'RMSE':>10} | {'MAPE (%)':>9}")
print("-" * 62)
for name, (tr, te) in res.items():
    print(f"{name:<16} | {'train':<5} | {tr[0]:>10.4f} | {tr[1]:>10.4f} | {tr[2]:>9.4f}")
    print(f"{'':<16} | {'test':<5} | {te[0]:>10.4f} | {te[1]:>10.4f} | {te[2]:>9.4f}")

# ---------- Hình 1: tuyến tính hóa + đường cong hàm mũ ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
xs = np.linspace(1, 12, 200)

ax[0].scatter(x_train, np.log(y_train), color="green", label=r"$\ln y_i$ (train)")
ax[0].plot(xs, ln_a + b * xs, "g-", label=r"$\ln y = \ln a + bx$")
ax[0].set_xlabel("Tháng"); ax[0].set_ylabel(r"$\ln y$")
ax[0].set_title("Sau khi tuyến tính hóa bằng logarit")
ax[0].grid(alpha=0.3); ax[0].legend()

ax[1].scatter(x_train, y_train, color="green", label="Train")
ax[1].scatter(x_test, y_test, color="red", marker="s", label="Test")
ax[1].plot(xs, predict_exp(xs, a, b), "g-", label=r"$y=ae^{bx}$")
ax[1].set_xlabel("Tháng"); ax[1].set_ylabel("Doanh số")
ax[1].set_title("Xấp xỉ hàm mũ trên dữ liệu doanh số")
ax[1].grid(alpha=0.3); ax[1].legend()
plt.tight_layout()
plt.savefig("hinh_ham_mu.png", dpi=200)

# ---------- Hình 2: so sánh với đa thức ----------
plt.figure(figsize=(7, 4.5))
plt.scatter(x_train, y_train, color="green", label="Train")
plt.scatter(x_test, y_test, color="red", marker="s", label="Test")
plt.plot(xs, predict_exp(xs, a, b), "g-", lw=2, label="Hàm mũ")
for d, style in zip((1, 2, 3), ("b--", "m--", "k--")):
    plt.plot(xs, polys[d](xs), style, label=f"Đa thức bậc {d}")
plt.ylim(200, 1100)
plt.xlabel("Tháng"); plt.ylabel("Doanh số")
plt.title("So sánh hàm mũ và đa thức")
plt.grid(alpha=0.3); plt.legend()
plt.tight_layout()
plt.savefig("hinh_so_sanh_da_thuc.png", dpi=200)
plt.show()