from scipy import signal as scipySignal
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
#plt.show()

plt.figure()
fVec, Pxx_den = scipySignal.welch(signal_with_noise, fs, scaling='density', return_onesided=True)
plt.plot(fVec, watt2dbm(Pxx_den))
plt.xlabel('hz')
plt.ylabel('dbm / hz')
plt.grid(True)
plt.title('power spectral density')
#plt.show()

signalEstMaxPower_dbm = np.max(watt2dbm(Pxx_den))
noiseEstPower_dbm = watt2dbm(np.median(Pxx_den)) + watt2db(fVec.size)
# assumption (based on me knowing that the signal is sin): every bin with energy
# above noiseEstPower_dbm + 0.5*(signalEstMaxPower_dbm - noiseEstPower_dbm) contributes to the signals power
threshold = noiseEstPower_dbm + 0.5*(signalEstMaxPower_dbm - noiseEstPower_dbm)
print(f'threshold is at {threshold} dbm')
signalEstPower_dbm = watt2dbm(Pxx_den[np.where(watt2dbm(Pxx_den) > threshold)].sum())
estSNR_db = signalEstPower_dbm - noiseEstPower_dbm
print(f'Estimated signal power is {signalEstPower_dbm}')
print(f'Estimated noise power is {noiseEstPower_dbm}')
print(f'Estimated SNR is {estSNR_db}')


plt.show()
'''
nFft = 256
fftRes = fs/nFft
print(f'fft resolution is {fftRes} Hz')

nComplete_fftCycles = np.floor(tVec.size/nFft)
nSamplesForFft = int(nFft*nComplete_fftCycles)
overlapSig = np.sum(np.reshape(signal_with_noise[:nSamplesForFft], (-1, nFft)), axis=0)

sp_dbm = volt2dbm(np.abs(np.fft.fft(overlapSig)))
fVec = np.fft.fftfreq(nFft, 1/fs)

plt.plot(fVec, sp_dbm)
plt.xlabel('hz')
plt.ylabel('dbm')
plt.grid(True)
plt.show()
'''
