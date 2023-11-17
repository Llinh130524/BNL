import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft, istft, butter, lfilter
import yfinance as yf
from datetime import datetime
# Các thư viện cần cài đặt để chạy chương trình 
pip install pandas matplotlib numpy scipy yfinance git+https://github.com/ranaroussi/yfinance.git requests


# Hàm để lấy dữ liệu cổ phiếu theo thời gian thực
def get_real_time_data(symbol):
    # Tải dữ liệu cổ phiếu từ yfinance
    stock_data = yf.download(symbol, start="2022-01-01", end=datetime.today().strftime('%Y-%m-%d'))
    return stock_data['Close']

# Hàm để thực hiện STFT, lọc, và ngược lại STFT
def process_data(column_data, fs=1, nperseg=30, noverlap=15, lowcut=0.01, highcut=0.1):
    # Thực hiện STFT trên dữ liệu cổ phiếu
    f, t, Zxx = stft(column_data, fs, nperseg=nperseg, noverlap=noverlap)

    # Áp dụng bộ lọc thông qua
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    filtered_signal = lfilter(b, a, column_data)

    # Thực hiện STFT trên tín hiệu đã lọc
    f_filtered, t_filtered, Zxx_filtered = stft(filtered_signal, fs, nperseg=nperseg, noverlap=noverlap)
    #Biến đôi ngược để đưa data về miền time 
    t_inverse, signal_inverse = istft(Zxx_filtered, fs, nperseg=nperseg, noverlap=noverlap)

    return f, t, Zxx, filtered_signal, f_filtered, t_filtered, Zxx_filtered, t_inverse, signal_inverse

# Hàm để dự đoán giá cổ phiếu trong tương lai sử dụng LPC
def predict_future_prices(signal_inverse, order=10):
    a = np.poly(signal_inverse)
    predicted_signal = lfilter(a, [1.0], signal_inverse)
    return predicted_signal

# Chương trình chính
symbol = "GOOGL"
column_data_real_time = get_real_time_data(symbol)

# Xử lý dữ liệu
f, t, Zxx, filtered_signal, f_filtered, t_filtered, Zxx_filtered, t_inverse, signal_inverse = process_data(column_data_real_time)

# Vẽ đồ thị kết quả
plt.figure(figsize=(12, 16))

plt.subplot(4, 1, 1)
plt.plot(column_data_real_time)
plt.title("Dữ liệu cổ phiếu thời gian thực")

plt.subplot(4, 1, 2)
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
plt.title("STFT của dữ liệu cổ phiếu thời gian thực")

plt.subplot(4, 1, 3)
plt.plot(filtered_signal)
plt.title("Tín hiệu đã lọc")

plt.subplot(4, 1, 4)
predicted_signal = predict_future_prices(signal_inverse)
plt.plot(t_inverse[:len(predicted_signal)], predicted_signal)
plt.title("Dự đoán giá cổ phiếu sử dụng LPC")

plt.tight_layout()
plt.show()
