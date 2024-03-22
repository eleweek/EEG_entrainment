import matplotlib

def add_red_line_with_value(fig, value, delta_db):
    for ax in fig.axes:
        ax.axvline(x=value, color='red', linestyle='-', linewidth=1.0)

        y_min, y_max = ax.get_ylim()
        y_shift = (y_max - y_min) * 0.05  # Adjust the multiplication factor as needed

        offset = matplotlib.transforms.ScaledTranslation(2/72, 0, fig.dpi_scale_trans)
        text_transform = ax.transData + offset

        ax.text(value, y_max - y_shift, f'{value:.2f} Hz, {delta_db:.2f} dB',
                ha='left', va='top', color='red', fontsize=8, transform=text_transform)