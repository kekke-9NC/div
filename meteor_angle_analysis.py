import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
import meteor_sky_viewer as msv

def analyze_angles(info_files, radiant_ra, radiant_dec, output_path):
    """
    Analyzes the angle distribution between the direction from the radiant to the meteor start point
    and the meteor's actual path (start to end).
    
    Args:
        info_files (list): List of paths to meteor info.txt files.
        radiant_ra (float): Radiant Right Ascension in degrees.
        radiant_dec (float): Radiant Declination in degrees.
        output_path (str): Path to save the histogram image.
        
    Returns:
        tuple: (success (bool), message (str))
    """
    if not info_files:
        return False, "解析対象のファイルがありません。"

    angles = []
    valid_count = 0
    errors = 0

    try:
        radiant = SkyCoord(ra=radiant_ra*u.deg, dec=radiant_dec*u.deg)
    except Exception as e:
        return False, f"放射点の座標が無効です: {e}"

    for file_path in info_files:
        try:
            data = msv.parse_info_file(file_path)
            ra_start = msv.get_float(data, "RA Start (deg)")
            dec_start = msv.get_float(data, "Dec Start (deg)")
            ra_end = msv.get_float(data, "RA End (deg)")
            dec_end = msv.get_float(data, "Dec End (deg)")

            if None in (ra_start, dec_start, ra_end, dec_end):
                errors += 1
                continue

            # Create SkyCoord objects
            start = SkyCoord(ra=ra_start*u.deg, dec=dec_start*u.deg)
            end = SkyCoord(ra=ra_end*u.deg, dec=dec_end*u.deg)

            # Calculate Position Angles (East of North)
            # 1. Angle from Start point to Radiant
            pa_to_radiant = start.position_angle(radiant).deg
            
            # 2. Angle from Start point to End point
            pa_start_to_end = start.position_angle(end).deg

            # Calculate absolute difference between the two vectors
            diff = abs(pa_start_to_end - pa_to_radiant)
            
            # Normalize to [0, 180) to get the angle between lines
            diff = diff % 180
            
            # Get the acute angle (0 to 90)
            if diff > 90:
                diff = 180 - diff
            
            angles.append(diff)
            valid_count += 1

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            errors += 1

    if valid_count == 0:
        return False, "有効な流星データが見つかりませんでした。"

    # Plot Histogram
    try:
        plt.figure(figsize=(10, 6))
        # Plot histogram for acute angles (0 to 90 degrees) using 10-degree bins
        bin_edges = np.arange(0, 91, 10)  # [0,10,20,...,90]
        plt.hist(angles, bins=bin_edges, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title(f"Acute Angle Distribution relative to Radiant (RA:{radiant_ra:.1f}, Dec:{radiant_dec:.1f})")
        plt.xlabel("Acute Angle (deg) [0 = Parallel to Radiant Direction]")
        plt.ylabel("Count")
        plt.grid(axis='y', alpha=0.5)
        # Expected direction is 0 degrees (parallel)
        plt.axvline(0, color='red', linestyle='dashed', linewidth=1, label='Parallel')
        plt.legend()
        
        # Add stats
        mean_angle = np.mean(angles)
        std_angle = np.std(angles)
        stats_text = f"N={valid_count}\nMean={mean_angle:.1f}°\nStd={std_angle:.1f}°"
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', horizontalalignment='right', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.savefig(output_path)
        plt.close()
        
        return True, f"解析完了: {valid_count}個のデータを処理しました。\n保存先: {output_path}"
    except Exception as e:
        return False, f"グラフの作成に失敗しました: {e}"
