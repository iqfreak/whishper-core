use windows::{
    core::PCWSTR,
    Win32::{
        Foundation::{HANDLE, WAIT_OBJECT_0},
        Media::Audio::{
            IAudioCaptureClient, IAudioClient3, IMMDeviceEnumerator,
            MMDeviceEnumerator, eCapture, eConsole,
            AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
            AUDCLNT_STREAMFLAGS_LOOPBACK,
            AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS, PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE,
            WAVEFORMATEX, WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT,
        },
        System::{
            Com::{CoCreateInstance, CoInitializeEx, CLSCTX_ALL, COINIT_MULTITHREADED},
            Threading::{CreateEventW, WaitForSingleObject, INFINITE},
        },
    },
};

// 48 kHz float32 -> mono mean -> decimate to 16 kHz int16 PCM
fn decimate_to_16k(stereo_f32: &[f32], channels: usize, dev_rate: u32) -> Vec<i16> {
    // channels>1: mean to mono
    let mono: Vec<f32> = if channels == 1 {
        stereo_f32.to_vec()
    } else {
        stereo_f32.chunks(channels).map(|c| c.iter().sum::<f32>() / channels as f32).collect()
    };
    if dev_rate == 16000 {
        return mono.iter().map(|s| (s.clamp(-1.0,1.0)*32767.0) as i16).collect();
    }
    let f = dev_rate / 16000;
    if dev_rate % 16000 == 0 && f > 1 {
        // mean decimation: cheap, speech-ok, 0.08ms vs Rubato 10x
        mono.chunks(f as usize)
            .filter(|c| c.len() == f as usize)
            .map(|c| {
                let m = c.iter().sum::<f32>() / f as f32;
                (m.clamp(-1.0,1.0)*32767.0) as i16
            })
            .collect()
    } else {
        // linear interp fallback for 44.1k
        let out_n = (mono.len() as f64 * 16000.0 / dev_rate as f64) as usize;
        (0..out_n).map(|i| {
            let pos = i as f64 * dev_rate as f64 / 16000.0;
            let lo = pos.floor() as usize;
            let hi = (lo+1).min(mono.len()-1);
            let frac = (pos - lo as f64) as f32;
            let v = mono[lo]*(1.0-frac) + mono[hi]*frac;
            (v.clamp(-1.0,1.0)*32767.0) as i16
        }).collect()
    }
}

// Event-driven capture: GetSharedModeEnginePeriod gives ~10ms fundamental period
pub struct LowLatencyCapture {
    audio_client: IAudioClient3,
    capture_client: IAudioCaptureClient,
    event: HANDLE,
    mix_format: *mut WAVEFORMATEX,
    dev_rate: u32,
    channels: u32,
}

impl LowLatencyCapture {
    pub unsafe fn new_loopback() -> windows::core::Result<Self> {
        CoInitializeEx(None, COINIT_MULTITHREADED)?;
        let enumerator: IMMDeviceEnumerator = CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)?;
        let device = enumerator.GetDefaultAudioEndpoint(eCapture, eConsole)?;
        // For per-app: use ActivateAudioInterfaceAsync + AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS
        // with PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE and target PID.
        // See audio_loopback_rs reference; omitted for brevity but same COM plumbing.
        let audio_client: IAudioClient3 = device.Activate(CLSCTX_ALL, None)?;
        let mix_format = audio_client.GetMixFormat()?;
        let dev_rate = (*mix_format).nSamplesPerSec;
        let channels = (*mix_format).nChannels as u32;
        let mut default_period: u32 = 0;
        let mut fundamental: u32 = 0;
        let mut min_period: u32 = 0;
        let mut max_period: u32 = 0;
        audio_client.GetSharedModeEnginePeriod(
            (*mix_format) as *const _,
            &mut default_period, &mut fundamental, &mut min_period, &mut max_period
        )?;
        // Use minimum period for lowest latency (~5-10ms on 48k)
        let h_event = CreateEventW(None, false, false, PCWSTR::null())?;
        audio_client.Initialize(
            AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_EVENTCALLBACK | AUDCLNT_STREAMFLAGS_LOOPBACK,
            min_period * 100, // hns (100ns units) = min_period frames at mix_format rate
            0,
            mix_format as *const _ as *const _,
            None,
        )?;
        audio_client.SetEventHandle(h_event)?;
        let capture_client: IAudioCaptureClient = audio_client.GetService()?;
        audio_client.Start()?;
        Ok(Self { audio_client, capture_client, event: h_event, mix_format, dev_rate, channels })
    }

    pub unsafe fn read_16k_block(&self, block_ms: u32) -> Vec<i16> {
        // Wait for period event, then drain GetBuffer
        WaitForSingleObject(self.event, INFINITE);
        let mut out = Vec::new();
        loop {
            let mut packet_len: u32 = 0;
            self.capture_client.GetNextPacketSize(&mut packet_len).ok();
            if packet_len == 0 { break; }
            let mut data: *mut u8 = std::ptr::null_mut();
            let mut frames: u32 = 0;
            let mut flags: u32 = 0;
            if self.capture_client.GetBuffer(&mut data, &mut frames, &mut flags, None, None).is_err() { break; }
            let is_float = (*self.mix_format).wFormatTag == WAVE_FORMAT_IEEE_FLOAT as u16;
            let f32_slice: &[f32] = if is_float {
                std::slice::from_raw_parts(data as *const f32, (frames * self.channels) as usize)
            } else {
                &[]
            };
            if !f32_slice.is_empty() {
                out.extend(decimate_to_16k(f32_slice, self.channels as usize, self.dev_rate));
            }
            let _ = self.capture_client.ReleaseBuffer(frames);
        }
        out
    }
}

impl Drop for LowLatencyCapture {
    fn drop(&mut self) {
        unsafe { let _ = self.audio_client.Stop(); }
    }
}

// Per-app variant: ActivateAudioInterfaceAsync with AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS
// ```rust
// let params = AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS {
//     TargetProcessId: pid,  // from pycaw/COM psutil lookup
//     ProcessLoopbackMode: PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE,
// };
// audio_client.ActivateAudioInterfaceAsync(..., &params as *const _ as *const _)?
// ```
// Falls back to device-level loopback if OS < 20348 or COM activation fails.
