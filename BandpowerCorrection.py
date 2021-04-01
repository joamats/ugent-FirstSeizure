"""
This file contains several helper functions to calculate spectral power from
1D and 2D EEG data.
"""
import mne
import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import simps


# Modification from "bandpower" to include 1/f correction
def bandpower_1f_correction(data, sf=None, ch_names=None, hypno=None, include=(2, 3),
              win_sec=4, relative=True, bandpass=False,
              bands=[(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
                     (12, 16, 'Sigma'), (16, 30, 'Beta'), (30, 40, 'Gamma')],
              kwargs_welch=dict(average='median', window='hamming')):
    """
    Calculate the Welch bandpower for each channel and, if specified,
    for each sleep stage.

    .. versionadded:: 0.1.6

    Parameters
    ----------
    data : np.array_like or :py:class:`mne.io.BaseRaw`
        1D or 2D EEG data. Can also be a :py:class:`mne.io.BaseRaw`, in which
        case ``data``, ``sf``, and ``ch_names`` will be automatically
        extracted, and ``data`` will also be converted from Volts (MNE default)
        to micro-Volts (YASA).
    sf : float
        The sampling frequency of data AND the hypnogram.
        Can be omitted if ``data`` is a :py:class:`mne.io.BaseRaw`.
    ch_names : list
        List of channel names, e.g. ['Cz', 'F3', 'F4', ...]. If None,
        channels will be labelled ['CHAN000', 'CHAN001', ...].
        Can be omitted if ``data`` is a :py:class:`mne.io.BaseRaw`.
    hypno : array_like
        Sleep stage (hypnogram). If the hypnogram is loaded, the
        bandpower will be extracted for each sleep stage defined in
        ``include``.

        The hypnogram must have the exact same number of samples as ``data``.
        To upsample your hypnogram, please refer to
        :py:func:`yasa.hypno_upsample_to_data`.

        .. note::
            The default hypnogram format in YASA is a 1D integer
            vector where:

            - -2 = Unscored
            - -1 = Artefact / Movement
            - 0 = Wake
            - 1 = N1 sleep
            - 2 = N2 sleep
            - 3 = N3 sleep
            - 4 = REM sleep
    include : tuple, list or int
        Values in ``hypno`` that will be included in the mask. The default is
        (2, 3), meaning that the bandpower are sequentially calculated
        for N2 and N3 sleep. This has no effect when ``hypno`` is None.
    win_sec : int or float
        The length of the sliding window, in seconds, used for the Welch PSD
        calculation. Ideally, this should be at least two times the inverse of
        the lower frequency of interest (e.g. for a lower frequency of interest
        of 0.5 Hz, the window length should be at least 2 * 1 / 0.5 =
        4 seconds).
    relative : boolean
        If True, bandpower is divided by the total power between the min and
        max frequencies defined in ``band``.
    bandpass : boolean
        If True, apply a standard FIR bandpass filter using the minimum and
        maximum frequencies in ``bands``. Fore more details, refer to
        :py:func:`mne.filter.filter_data`.
    bands : list of tuples
        List of frequency bands of interests. Each tuple must contain the
        lower and upper frequencies, as well as the band name
        (e.g. (0.5, 4, 'Delta')).
    kwargs_welch : dict
        Optional keywords arguments that are passed to the
        :py:func:`scipy.signal.welch` function.

    Returns
    -------
    bandpowers : :py:class:`pandas.DataFrame`
        Bandpower dataframe, in which each row is a channel and each column
        a spectral band.

    Notes
    -----
    For an example of how to use this function, please refer to
    https://github.com/raphaelvallat/yasa/blob/master/notebooks/08_bandpower.ipynb
    """
    # Type checks
    assert isinstance(bands, list), 'bands must be a list of tuple(s)'
    assert isinstance(relative, bool), 'relative must be a boolean'
    assert isinstance(bandpass, bool), 'bandpass must be a boolean'

    # Check if input data is a MNE Raw object
    if isinstance(data, mne.io.BaseRaw):
        sf = data.info['sfreq']  # Extract sampling frequency
        ch_names = data.ch_names  # Extract channel names
        data = data.get_data() * 1e6  # Convert from V to uV
        _, npts = data.shape
    else:
        # Safety checks
        assert isinstance(data, np.ndarray), 'Data must be a numpy array.'
        data = np.atleast_2d(data)
        assert data.ndim == 2, 'Data must be of shape (nchan, n_samples).'
        nchan, npts = data.shape
        # assert nchan < npts, 'Data must be of shape (nchan, n_samples).'
        assert sf is not None, 'sf must be specified if passing a numpy array.'
        assert isinstance(sf, (int, float))
        if ch_names is None:
            ch_names = ['CHAN' + str(i).zfill(3) for i in range(nchan)]
        else:
            ch_names = np.atleast_1d(np.asarray(ch_names, dtype=str))
            assert ch_names.ndim == 1, 'ch_names must be 1D.'
            assert len(ch_names) == nchan, 'ch_names must match data.shape[0].'

    if bandpass:
        # Apply FIR bandpass filter
        all_freqs = np.hstack([[b[0], b[1]] for b in bands])
        fmin, fmax = min(all_freqs), max(all_freqs)
        data = mne.filter.filter_data(data.astype('float64'), sf, fmin, fmax,
                                      verbose=0)

    win = int(win_sec * sf)  # nperseg

    if hypno is None:
        # Calculate the PSD over the whole data
        freqs, psd = signal.welch(data, sf, nperseg=win, **kwargs_welch)
        
        for i in range(np.shape(psd)[0]):
            psd[i,:] *= freqs
        
        return bandpower_from_psd(psd, freqs, ch_names, bands=bands,
                                  relative=relative).set_index('Chan'), freqs, psd
    else:
        # Per each sleep stage defined in ``include``.
        hypno = np.asarray(hypno)
        assert include is not None, 'include cannot be None if hypno is given'
        include = np.atleast_1d(np.asarray(include))
        assert hypno.ndim == 1, 'Hypno must be a 1D array.'
        assert hypno.size == npts, 'Hypno must have same size as data.shape[1]'
        assert include.size >= 1, '`include` must have at least one element.'
        assert hypno.dtype.kind == include.dtype.kind, ('hypno and include '
                                                        'must have same dtype')
        assert np.in1d(hypno, include).any(), ('None of the stages '
                                               'specified in `include` '
                                               'are present in hypno.')
        # Initialize empty dataframe and loop over stages
        df_bp = pd.DataFrame([])
        for stage in include:
            if stage not in hypno:
                continue
            data_stage = data[:, hypno == stage]
            freqs, psd = signal.welch(data_stage, sf, nperseg=win,
                                      **kwargs_welch)
            
            bp_stage = bandpower_from_psd(psd, freqs, ch_names, bands=bands,
                                          relative=relative)
            bp_stage['Stage'] = stage
            df_bp = df_bp.append(bp_stage)
            
        return df_bp.set_index(['Stage', 'Chan'])


def bandpower_from_psd(psd, freqs, ch_names=None, bands=[(0.5, 4, 'Delta'),
                       (4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 16, 'Sigma'),
                       (16, 30, 'Beta'), (30, 40, 'Gamma')], relative=True):
    """Compute the average power of the EEG in specified frequency band(s)
    given a pre-computed PSD.

    .. versionadded:: 0.1.5

    Parameters
    ----------
    psd : array_like
        Power spectral density of data, in uV^2/Hz.
        Must be of shape (n_channels, n_freqs).
        See :py:func:`scipy.signal.welch` for more details.
    freqs : array_like
        Array of frequencies.
    ch_names : list
        List of channel names, e.g. ['Cz', 'F3', 'F4', ...]. If None,
        channels will be labelled ['CHAN000', 'CHAN001', ...].
    bands : list of tuples
        List of frequency bands of interests. Each tuple must contain the
        lower and upper frequencies, as well as the band name
        (e.g. (0.5, 4, 'Delta')).
    relative : boolean
        If True, bandpower is divided by the total power between the min and
        max frequencies defined in ``band`` (default 0.5 to 40 Hz).

    Returns
    -------
    bandpowers : :py:class:`pandas.DataFrame`
        Bandpower dataframe, in which each row is a channel and each column
        a spectral band.
    """
    # Type checks
    assert isinstance(bands, list), 'bands must be a list of tuple(s)'
    assert isinstance(relative, bool), 'relative must be a boolean'

    # Safety checks
    freqs = np.asarray(freqs)
    assert freqs.ndim == 1
    psd = np.atleast_2d(psd)
    assert psd.ndim == 2, 'PSD must be of shape (n_channels, n_freqs).'
    all_freqs = np.hstack([[b[0], b[1]] for b in bands])
    fmin, fmax = min(all_freqs), max(all_freqs)
    idx_good_freq = np.logical_and(freqs >= fmin, freqs <= fmax)
    freqs = freqs[idx_good_freq]
    res = freqs[1] - freqs[0]
    nchan = psd.shape[0]
    assert nchan < psd.shape[1], 'PSD must be of shape (n_channels, n_freqs).'
    if ch_names is not None:
        ch_names = np.atleast_1d(np.asarray(ch_names, dtype=str))
        assert ch_names.ndim == 1, 'ch_names must be 1D.'
        assert len(ch_names) == nchan, 'ch_names must match psd.shape[0].'
    else:
        ch_names = ['CHAN' + str(i).zfill(3) for i in range(nchan)]
    bp = np.zeros((nchan, len(bands)), dtype=np.float)
    psd = psd[:, idx_good_freq]
    total_power = simps(psd, dx=res)
    total_power = total_power[..., np.newaxis]

    # Enumerate over the frequency bands
    labels = []
    for i, band in enumerate(bands):
        b0, b1, la = band
        labels.append(la)
        idx_band = np.logical_and(freqs >= b0, freqs <= b1)
        bp[:, i] = simps(psd[:, idx_band], dx=res)

    if relative:
        bp /= total_power

    # Convert to DataFrame
    bp = pd.DataFrame(bp, columns=labels)
    bp['TotalAbsPow'] = np.squeeze(total_power)
    bp['FreqRes'] = res
    # bp['WindowSec'] = 1 / res
    bp['Relative'] = relative
    bp['Chan'] = ch_names
    bp = bp.set_index('Chan').reset_index()
    # Add hidden attributes
    bp.bands_ = str(bands)
    return bp
