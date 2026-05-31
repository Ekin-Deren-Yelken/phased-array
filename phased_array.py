import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, RadioButtons

# Physical parameters
c = 1500.0              # speed of sound in water [m/s]
frequency = 100e3      # Hz
wavelength = c / frequency
omega = 2 * np.pi * frequency
k = 2 * np.pi / wavelength

# Underwater acoustic pressure reference
p0 = 1e-6               # 1 microPascal

# Simulation grid, in wavelengths
x_min, x_max = -50, 50
y_min, y_max = 0.25, 100
resolution = 700

x = np.linspace(x_min * wavelength, x_max * wavelength, resolution)
y = np.linspace(y_min * wavelength, y_max * wavelength, resolution)
X, Y = np.meshgrid(x, y)

# Initial array settings
initial_N = 2
initial_transducer_spacing = 0.5       # this is d/lambda. Optimal is d = lambda / 2, thereform d/lambda = 1/2 = 0.5
initial_phase_step_deg = 0.0
initial_decay_power = 0.5          # 0=no decay, 0.5=1/sqrt(R), 1=1/R
initial_db_floor = -30.0
initial_pressure_contrast = 1.0
initial_mode = "pressure"
initial_directivity_mode = "directivity dB"

def compute_phasor(num_elements, transducer_space, phase_step_deg, decay_power):
    """
    Compute complex pressure phasor P(x,y), where:

        p(x,y,t) = Re{ P(x,y) exp(-j omega t) }

    decay_power controls visual/physical spreading:
        0.0 -> no decay
        0.5 -> 1/sqrt(R)
        1.0 -> 1/R
    """
    spacing = transducer_space * wavelength
    phase_step = np.deg2rad(phase_step_deg)

    element_indices = np.arange(num_elements) - (num_elements - 1) / 2
    element_x = element_indices * spacing
    element_y = np.zeros_like(element_x)

    P = np.zeros_like(X, dtype=np.complex128)

    for n, (tx, ty) in enumerate(zip(element_x, element_y)):
        R = np.sqrt((X - tx) ** 2 + (Y - ty) ** 2)
        R = np.maximum(R, 0.05 * wavelength)  # avoid singularity at source

        phase = n * phase_step

        if decay_power <= 1e-9:
            spreading = 1.0
        else:
            spreading = R ** decay_power

        P += np.exp(1j * (k * R + phase)) / spreading

    P /= np.max(np.abs(P)) + 1e-15
    return P, element_x / wavelength

def compute_directivity(num_elements, transducer_space, phase_step_deg, db_floor):
    """
    Far-field array factor for a uniform linear array.

    Angle convention:
        theta = 0       broadside, +y direction
        theta = +/-90   endfire, along +/-x
        theta = +/-180  rear broadside, -y direction

    Sign convention:
        source phase term is +n*beta,
        far-field path phase is -k*x_n*sin(theta).
    """
    d = transducer_space * wavelength
    beta = np.deg2rad(phase_step_deg)

    theta = np.linspace(-np.pi, np.pi, 2400)
    n = np.arange(num_elements) - (num_elements - 1) / 2

    AF = np.zeros_like(theta, dtype=np.complex128)
    for ni in n:
        AF += np.exp(1j * ni * (beta - k * d * np.sin(theta)))

    AF_mag = np.abs(AF)
    AF_mag /= np.max(AF_mag) + 1e-15

    AF_db = 20 * np.log10(AF_mag + 1e-12)
    AF_db = np.clip(AF_db, db_floor, 0.0)

    # Polar plots cannot show negative radius cleanly, so map:
    # db_floor -> 0, 0 dB -> abs(db_floor)
    AF_db_radius = AF_db - db_floor

    return theta, AF_mag, AF_db, AF_db_radius

def field_to_image(P, t, mode, db_floor, pressure_contrast):
    if mode == "pressure":
        p_inst = np.real(P * np.exp(-1j * omega * t))
        img = np.tanh(pressure_contrast * p_inst)
        return img, -1, 1, "Instantaneous pressure"

    if mode == "intensity dB":
        mean_p2 = np.abs(P) ** 2 / 2.0
        intensity_db = 10 * np.log10(mean_p2 / (np.max(mean_p2) + 1e-15) + 1e-12)
        intensity_db = np.clip(intensity_db, db_floor, 0.0)
        return intensity_db, db_floor, 0, "Relative intensity [dB]"

    if mode == "SPL dB re 1 µPa":
        # Because P is normalized, this is display SPL, not calibrated absolute SPL.
        mean_p2 = np.abs(P) ** 2 / 2.0
        spl = 10 * np.log10(mean_p2 / p0**2 + 1e-30)
        return spl, np.max(spl) + db_floor, np.max(spl), "SPL [dB re 1 µPa]"

    raise ValueError(f"Unknown mode: {mode}")

# Build initial data
P, element_positions = compute_phasor(initial_N, initial_transducer_spacing, initial_phase_step_deg, initial_decay_power)

theta0, AF_mag0, AF_db0, AF_db_radius0 = compute_directivity(initial_N, initial_transducer_spacing, initial_phase_step_deg, initial_db_floor)

current = {
    "P": P,
    "positions": element_positions,
    "mode": initial_mode,
    "directivity_mode": initial_directivity_mode,
    "db_floor": initial_db_floor,
    "pressure_contrast": initial_pressure_contrast,
    "theta": theta0,
    "AF_mag": AF_mag0,
    "AF_db": AF_db0,
    "AF_db_radius": AF_db_radius0,
}

# Figure layout
fig = plt.figure(figsize=(15, 8))

ax_field = fig.add_axes([0.07, 0.36, 0.50, 0.58])
ax_polar = fig.add_axes([0.64, 0.48, 0.28, 0.38], projection="polar")
ax_cart = fig.add_axes([0.63, 0.35, 0.31, 0.12])

image_data, vmin, vmax, title = field_to_image(
    current["P"], 0.0, current["mode"], current["db_floor"], current["pressure_contrast"]
)

img = ax_field.imshow(
    image_data,
    extent=[x_min, x_max, y_min, y_max],
    origin="lower",
    aspect="auto",
    cmap="gist_gray",
    vmin=vmin,
    vmax=vmax,
)

scatter = ax_field.scatter(
    current["positions"],
    np.zeros_like(current["positions"]),
    marker=".",
    s=180,
)

ax_field.set_xlabel("x position [wavelengths]")
ax_field.set_ylabel("range y [wavelengths]")
ax_field.set_title(title)

cbar = plt.colorbar(img, ax=ax_field)
cbar.set_label(title)

# Polar directivity: dB by default
polar_line, = ax_polar.plot(current["theta"], current["AF_db_radius"])
ax_polar.set_title("Azimuthal directivity")
ax_polar.set_theta_zero_location("N")
ax_polar.set_theta_direction(-1)
ax_polar.set_rlim(0, abs(initial_db_floor))
ax_polar.set_yticks([0, abs(initial_db_floor) / 2, abs(initial_db_floor)])
ax_polar.set_yticklabels([f"{initial_db_floor:.0f}", f"{initial_db_floor / 2:.0f}", "0"])

# Cartesian directivity in dB vs angle
angle_deg = np.rad2deg(current["theta"])
cart_line, = ax_cart.plot(angle_deg, current["AF_db"])
ax_cart.set_xlim(-180, 180)
ax_cart.set_ylim(initial_db_floor, 0)
ax_cart.set_xlabel("angle from broadside [deg]")
ax_cart.set_ylabel("dB")
ax_cart.grid(True, alpha=0.3)

# Widgets
ax_N = plt.axes([0.15, 0.25, 0.4, 0.03])
ax_spacing = plt.axes([0.15, 0.20, 0.4, 0.03])
ax_phase = plt.axes([0.15, 0.15, 0.4, 0.03])
ax_decay = plt.axes([0.15, 0.10, 0.4, 0.03])
ax_dbfloor = plt.axes([0.15, 0.05, 0.4, 0.03])

slider_N = Slider(ax_N, "Elements", 1, 32, valinit=initial_N, valstep=1)
slider_spacing = Slider(ax_spacing, "Spacing d/λ", 0.1, 4.0, valinit=initial_transducer_spacing)
slider_phase = Slider(ax_phase, "Phase step [deg]", -180, 180, valinit=initial_phase_step_deg)
slider_decay = Slider(ax_decay, "Decay power", 0.0, 1.0, valinit=initial_decay_power)
slider_dbfloor = Slider(ax_dbfloor, "dB floor", -80.0, -5.0, valinit=initial_db_floor)

ax_radio_field = plt.axes([0.78, 0.12, 0.17, 0.14])
radio_field = RadioButtons(
    ax_radio_field,
    ("pressure", "intensity dB", "SPL dB re 1 µPa"),
    active=0,
)

ax_radio_dir = plt.axes([0.64, 0.12, 0.12, 0.10])
radio_dir = RadioButtons(
    ax_radio_dir,
    ("directivity dB", "directivity linear"),
    active=0,
)

def update_directivity_axes():
    if current["directivity_mode"] == "directivity linear":
        polar_line.set_data(current["theta"], current["AF_mag"])
        ax_polar.set_rlim(0, 1.0)
        ax_polar.set_yticks([0.0, 0.5, 1.0])
        ax_polar.set_yticklabels(["0", "0.5", "1"])
    else:
        floor = current["db_floor"]
        polar_line.set_data(current["theta"], current["AF_db_radius"])
        ax_polar.set_rlim(0, abs(floor))
        ax_polar.set_yticks([0, abs(floor) / 2, abs(floor)])
        ax_polar.set_yticklabels([f"{floor:.0f}", f"{floor / 2:.0f}", "0"])

    cart_line.set_data(np.rad2deg(current["theta"]), current["AF_db"])
    ax_cart.set_ylim(current["db_floor"], 0)

def recompute(_=None):
    N = int(slider_N.val)
    transducer_space = slider_spacing.val
    phase_step_deg = slider_phase.val
    decay_power = slider_decay.val
    db_floor = slider_dbfloor.val

    P, positions = compute_phasor(N, transducer_space, phase_step_deg, decay_power)
    theta, AF_mag, AF_db, AF_db_radius = compute_directivity(
        N, transducer_space, phase_step_deg, db_floor
    )

    current["P"] = P
    current["positions"] = positions
    current["db_floor"] = db_floor
    current["theta"] = theta
    current["AF_mag"] = AF_mag
    current["AF_db"] = AF_db
    current["AF_db_radius"] = AF_db_radius

    scatter.set_offsets(np.column_stack([positions, np.zeros_like(positions)]))
    update_directivity_axes()
    fig.canvas.draw_idle()

def mode_update(label):
    current["mode"] = label
    fig.canvas.draw_idle()

def directivity_mode_update(label):
    current["directivity_mode"] = label
    update_directivity_axes()
    fig.canvas.draw_idle()

slider_N.on_changed(recompute)
slider_spacing.on_changed(recompute)
slider_phase.on_changed(recompute)
slider_decay.on_changed(recompute)
slider_dbfloor.on_changed(recompute)
radio_field.on_clicked(mode_update)
radio_dir.on_clicked(directivity_mode_update)

def animate(frame):
    t = frame / 40.0 / frequency

    data, vmin, vmax, title = field_to_image(
        current["P"],
        t,
        current["mode"],
        current["db_floor"],
        current["pressure_contrast"],
    )

    img.set_data(data)
    img.set_clim(vmin, vmax)
    ax_field.set_title(title)
    cbar.set_label(title)

    return [img, scatter, polar_line, cart_line]

ani = FuncAnimation(fig, animate, frames=200, interval=30, blit=False)
plt.show()
