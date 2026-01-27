"""
Date, time, and data formatting utilities
"""

from datetime import datetime


def format_dicom_date(date_str: str) -> str:
    """Format DICOM date (YYYYMMDD) to YYYY-MM-DD.
    
    Args:
        date_str: DICOM date string in YYYYMMDD format
        
    Returns:
        Formatted date as YYYY-MM-DD or 'nan' if invalid
    """
    if date_str == 'nan' or not date_str or len(date_str) != 8:
        return 'nan'
    try:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    except:
        return 'nan'


def format_dicom_time(time_str: str) -> str:
    """Format DICOM time (HHMMSS) to HH:MM:SS.
    
    Args:
        time_str: DICOM time string in HHMMSS format
        
    Returns:
        Formatted time as HH:MM:SS or 'nan' if invalid
    """
    if time_str == 'nan' or not time_str or len(time_str) < 6:
        return 'nan'
    try:
        return f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
    except:
        return 'nan'


def calculate_age(dob_str: str, dos_str: str) -> float:
    """Calculate age in years at time of scan.
    
    Args:
        dob_str: Date of birth in YYYY-MM-DD format
        dos_str: Date of scan in YYYY-MM-DD format
        
    Returns:
        Age in years as float or 'nan' if invalid
    """
    try:
        dob = datetime.strptime(dob_str, '%Y-%m-%d')
        dos = datetime.strptime(dos_str, '%Y-%m-%d')
        age_days = (dos - dob).days
        return round(age_days / 365.25, 1)
    except:
        return 'nan'


def normalize_class_label(label: str) -> str:
    """Normalize class label to standard format.
    
    Args:
        label: Raw class label
        
    Returns:
        Normalized class label (A, B, C, D, AB, AC, AD, BC, BD, NG)
    """
    label = label.strip().upper()
    # Replace + with nothing for consistency (B+D -> BD)
    label = label.replace('+', '')
    # Valid labels
    valid_labels = {'A', 'B', 'C', 'D', 'AB', 'AC', 'AD', 'BC', 'BD', 'NG'}
    return label if label in valid_labels else 'NG'
