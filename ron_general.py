import numpy as np
import matplotlib.pyplot as plt


def dbm2var(x_dbm):
    return np.power(10, np.divide(x_dbm - 30, 10))

def dbm2std(x_dbm):
    return np.sqrt(np.power(10, np.divide(x_dbm - 30, 10)))

def volt2dbm(x_volt):
    return 10*np.log10(np.power(x_volt, 2)) + 30

def volt2dbW(x_volt):
    return 10*np.log10(np.power(x_volt, 2))

def volt2db(x_volt):
    return 10*np.log10(np.power(x_volt, 2))

def watt2dbm(x_volt):
    return 10*np.log10(x_volt) + 30

def watt2db(x_volt):
    return 10*np.log10(x_volt)

# general params:
fs = 500  # [hz]
nSamples = 10*fs
tVec = np.arange(0, nSamples) / fs
SNR = 24  # [db]

# signal params:
fc = 60  # [hz]
signalPower_dbm = 0
signalPhase = np.random.rand()*(2*np.pi)
signalVolt = dbm2std(signalPower_dbm)
omega = 2*np.pi*fc

signal = signalVolt*np.sin(omega*tVec + signalPhase)
print(f'{signal.shape}')

# noise params:
noisePower_dbm = signalPower_dbm - SNR
noiseStd = dbm2std(noisePower_dbm)
noise = noiseStd * np.random.randn(tVec.size)
print(f'{noise.shape}')

signal_with_noise = signal + noise
print(f'{signal_with_noise.shape}')

np.save('testSig.npy', signal_with_noise)

plt.plot(tVec, volt2dbm(signal_with_noise), label='signal&noise')
plt.plot(tVec, volt2dbm(signal), label='signal')
plt.plot(tVec, volt2dbm(noise), label='noise')
plt.xlabel('sec')
plt.ylabel('dbm')
plt.legend()
plt.grid(True)
plt.show()
