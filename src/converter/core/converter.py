"""
Main Universal DICOM to NIfTI Converter
Orchestrates the entire conversion pipeline
"""

import os
import csv
import json
import SimpleITK as sitk
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

from .dataset_detector import DatasetDetector
from .protocol_detector import ProtocolDetector
from .scan_plane_detector import ScanPlaneDetector
from .metadata_extractor import MetadataExtractor
from .validator import ConversionValidator
from ..utils.dicom_utils import read_dicom_series, read_dicom_series_with_quality_check, convert_to_ras, embed_metadata
from ..utils.formatting import normalize_class_label


class UniversalDICOMConverter:
    """Universal DICOM to NIfTI converter with automatic dataset detection."""
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        classification_csv: Optional[str] = None,
        start_case_id: int = 1
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.classification_csv = Path(classification_csv) if classification_csv else None
        self.start_case_id = start_case_id
        
        # Internal state
        self.dataset_type: str = 'generic'
        self.class_mapping: Dict[str, str] = {}
        self.csv_case_mapping: Dict[str, str] = {}
        self.patient_to_case: Dict[str, str] = {}
        self.case_counter: int = start_case_id - 1
        
        # Conversion tracking
        self.conversion_records: List[Dict] = []
        self.failed_cases: List[Dict] = []
        self.skipped_series: List[Dict] = []
        self.class_distribution: Dict[str, int] = defaultdict(int)
        self.protocol_distribution: Dict[str, int] = defaultdict(int)
        self.scan_plane_distribution: Dict[str, int] = defaultdict(int)
        self.modality_distribution: Dict[str, int] = defaultdict(int)
        self.quality_issues: Dict[str, int] = defaultdict(int)
        self.patient_id_sources: Dict[str, int] = defaultdict(int)
    
    def run(self) -> Dict:
        self._print_header()
        
        print("[1/6] Detecting dataset type...")
        self.dataset_type = DatasetDetector.detect_dataset_type(self.input_dir)
        print(f"      Detected: {self.dataset_type.upper()}")
        
        print("\n[2/6] Loading classifications...")
        self._load_classifications()
        print(f"      Loaded {len(self.class_mapping)} patient classifications")
        
        print("\n[3/6] Discovering DICOM series...")
        patient_series = self._discover_patient_series()
        print(f"      Found {len(patient_series)} patients with {sum(len(s) for s in patient_series.values())} series")
        
        print("\n[4/6] Converting DICOM to NIfTI...")
        for patient_id, series_list in patient_series.items():
            self._process_patient(patient_id, series_list)
        
        print("\n[5/6] Generating conversion summary...")
        summary = self._generate_summary()
        
        print("\n[6/6] Saving summary and manifest...")
        self._save_outputs(summary)
        
        self._print_completion(summary)
        return summary
    
    def _print_header(self):
        print("\n" + "="*80)
        print("UNIVERSAL DICOM TO NIFTI CONVERTER")
        print("="*80)
        print(f"Input directory:  {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Classification:   {self.classification_csv if self.classification_csv else 'None (all NG)'}")
        print("="*80 + "\n")
    
    def _print_completion(self, summary: Dict):
        print("\n" + "="*80)
        print("CONVERSION COMPLETE")
        print(f"Successful: {summary['successful_conversions']}")
        print(f"Failed:     {summary['failed_conversions']}")
        print("="*80 + "\n")
    
    def _load_classifications(self):
        if not self.classification_csv or not self.classification_csv.exists():
            print(f"      No classification CSV provided - all cases will be marked as 'NG'")
            return
        
        try:
            with open(self.classification_csv, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            headers = rows[0].keys() if rows else []
            patient_col = self._find_column(headers, ['TCGA ID', 'patient_id', 'PatientID', 'ID'])
            class_col = self._find_column(headers, ['Vessel evaluation', 'class', 'Class', 'Label'])
            case_col = self._find_column(headers, ['case_id', 'CaseID', 'Case'])
            
            if not patient_col or not class_col:
                print(f"      Warning: Could not find required columns in CSV")
                return
            
            for row in rows:
                patient_id = row[patient_col].strip()
                class_label = normalize_class_label(row[class_col])
                self.class_mapping[patient_id] = class_label
                
                if case_col and row.get(case_col):
                    self.csv_case_mapping[patient_id] = row[case_col].strip()
                    
            print(f"      Loaded {len(self.class_mapping)} classifications from CSV")
            
        except Exception as e:
            print(f"      Warning: Error loading classifications: {e}")
    
    def _find_column(self, headers, candidates: List[str]) -> Optional[str]:
        for candidate in candidates:
            if candidate in headers:
                return candidate
        return None
    
    def _discover_patient_series(self) -> Dict[str, List]:
        patient_series = defaultdict(list)
        print(f"      Discovering DICOM series by SeriesInstanceUID...")
        
        dicom_dirs = []
        for root, dirs, files in os.walk(self.input_dir):
            if files and not dirs:
                dicom_dirs.append(root)
        
        print(f"      Scanning {len(dicom_dirs)} directories for DICOM series...")
        
        series_found = []
        for search_dir in dicom_dirs:
            try:
                reader = sitk.ImageSeriesReader()
                series_uids = reader.GetGDCMSeriesIDs(str(search_dir))
                
                for series_uid in series_uids:
                    dicom_files = reader.GetGDCMSeriesFileNames(str(search_dir), series_uid)
                    if dicom_files:
                        series_found.append({
                            'series_uid': series_uid,
                            'directory': search_dir,
                            'files': dicom_files
                        })
            except Exception as e:
                continue
        
        print(f"      Found {len(series_found)} unique DICOM series (by SeriesInstanceUID)")
        
        if len(series_found) == 0:
            print(f"      WARNING: No DICOM series found. Check:")
            print(f"              - Input directory exists: {self.input_dir.exists()}")
            print(f"              - Contains .dcm files: {len(dicom_dirs)} dirs scanned")
            return patient_series
        
        for series_info in series_found:
            try:
                result = read_dicom_series_with_quality_check(series_info['files'])
                if result is None:
                    self.failed_cases.append({
                        'series_uid': series_info.get('series_uid', 'unknown'),
                        'series_path': series_info['directory'],
                        'error': 'Failed to read DICOM series',
                        'stage': 'discovery'
                    })
                    continue
                
                reader, image, quality_info = result
                patient_id, id_source = DatasetDetector.extract_patient_id(reader, series_info['directory'])
                self.patient_id_sources[id_source] += 1
                
                if patient_id == 'UNKNOWN':
                    print(f"      Warning: Could not extract patient ID from {series_info['directory']}")
                
                try:
                    modality = reader.GetMetaData(0, '0008|0060') if reader.HasMetaDataKey(0, '0008|0060') else 'nan'
                except:
                    modality = 'nan'
                
                if modality == 'SEG':
                    self.skipped_series.append({
                        'patient_id': patient_id,
                        'series_uid': series_info['series_uid'],
                        'series_path': series_info['directory'],
                        'modality': modality,
                        'reason': 'SEG modality (segmentation) - skipped',
                        'stage': 'filtering'
                    })
                    continue
                
                self.quality_issues[quality_info['severity']] += 1
                if modality != 'nan':
                    self.modality_distribution[modality] += 1
                
                if quality_info['severity'] == 'critical':
                    self.skipped_series.append({
                        'patient_id': patient_id,
                        'series_uid': series_info['series_uid'],
                        'series_path': series_info['directory'],
                        'modality': modality,
                        'nonuniformity_mm': quality_info['nonuniformity_mm'],
                        'reason': f'Critical spacing nonuniformity: {quality_info["nonuniformity_mm"]:.2f}mm',
                        'stage': 'quality_validation'
                    })
                    continue
                
                scan_plane = ScanPlaneDetector.determine_scan_plane(reader, image)
                self.scan_plane_distribution[scan_plane] += 1
                
                if scan_plane != 'AXIAL':
                    self.skipped_series.append({
                        'patient_id': patient_id,
                        'series_uid': series_info['series_uid'],
                        'series_path': series_info['directory'],
                        'modality': modality,
                        'scan_plane': scan_plane,
                        'reason': f'Non-axial scan ({scan_plane})',
                        'stage': 'filtering'
                    })
                    continue
                
                num_slices = len(series_info['files'])
                if num_slices < 10:
                    self.skipped_series.append({
                        'patient_id': patient_id,
                        'series_uid': series_info['series_uid'],
                        'series_path': series_info['directory'],
                        'modality': modality,
                        'scan_plane': scan_plane,
                        'num_slices': num_slices,
                        'reason': f'Too few slices ({num_slices} < 10) - likely scout/localizer',
                        'stage': 'filtering'
                    })
                    continue
                
                patient_series[patient_id].append({
                    'path': series_info['directory'],
                    'series_uid': series_info['series_uid'],
                    'image': image,
                    'dicom_files': series_info['files'],
                    'reader': reader,
                    'modality': modality,
                    'quality_info': quality_info
                })
                
            except Exception as e:
                self.failed_cases.append({
                    'series_uid': series_info.get('series_uid', 'unknown'),
                    'series_path': series_info['directory'],
                    'error': str(e),
                    'stage': 'discovery'
                })
                continue
        
        return patient_series
    
    def _process_patient(self, patient_id: str, series_list: List):
        class_label = self.class_mapping.get(patient_id, 'NG')
        case_id = self._assign_case_id(patient_id)
        class_dir = self.output_dir / class_label
        class_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"      Processing {patient_id} (class: {class_label}, case: {case_id}, {len(series_list)} series)")
        
        for scan_idx, series_info in enumerate(series_list):
            try:
                self._convert_series(series_info, patient_id, scan_idx, class_label, case_id, class_dir)
            except Exception as e:
                self.failed_cases.append({
                    'patient_id': patient_id,
                    'series_path': series_info['path'],
                    'error': str(e),
                    'stage': 'conversion'
                })
    
    def _assign_case_id(self, patient_id: str) -> str:
        if patient_id in self.csv_case_mapping:
            return self.csv_case_mapping[patient_id]
        if patient_id in self.patient_to_case:
            return self.patient_to_case[patient_id]
        
        self.case_counter += 1
        case_id = f"case_{self.case_counter:05d}"
        self.patient_to_case[patient_id] = case_id
        return case_id
    
    def _convert_series(self, series_info: Dict, patient_id: str, scan_idx: int,
                       class_label: str, case_id: str, class_dir: Path):
        image = series_info['image']
        reader = series_info['reader']
        dicom_files = series_info['dicom_files']
        modality = series_info.get('modality', 'nan')
        quality_info = series_info.get('quality_info', {'nonuniformity_mm': None, 'severity': 'ok'})
        
        ras_image = convert_to_ras(image)
        protocol, protocol_source = ProtocolDetector.determine_protocol(reader, series_info['path'])
        scan_plane = ScanPlaneDetector.determine_scan_plane(reader, image)
        metadata = MetadataExtractor.extract_comprehensive_metadata(reader, image, {
            'patient_id': patient_id,
            'class_label': class_label,
            'scan_idx': scan_idx,
            'protocol': protocol,
            'protocol_source': protocol_source,
            'scan_plane': scan_plane,
            'case_id': case_id,
            'dataset_type': self.dataset_type,
            'modality': modality,
            'spacing_quality': quality_info['severity'],
            'nonuniformity_mm': quality_info['nonuniformity_mm']
        })
        
        # Embed metadata in NIfTI
        ras_image = embed_metadata(ras_image, reader, metadata)
        
        # Build simplified filename
        filename = f"{scan_idx:02d}_{case_id}_0000.nii.gz"
        output_path = class_dir / filename
        
        # Save NIfTI with compression
        sitk.WriteImage(ras_image, str(output_path), True)
        
        # Validate conversion
        validation = ConversionValidator.validate_conversion(output_path, image)
        
        # Record conversion
        self._record_conversion(filename, patient_id, case_id, class_label, scan_idx,
                               protocol, protocol_source, scan_plane, metadata, 
                               image, dicom_files, output_path, validation)
        
        print(f"        ✓ Saved: {filename}")
    
    def _record_conversion(self, filename: str, patient_id: str, case_id: str,
                          class_label: str, scan_idx: int, protocol: str,
                          protocol_source: str, scan_plane: str, metadata: Dict,
                          image: sitk.Image, dicom_files: List, output_path: Path,
                          validation: Dict):
        spacing = image.GetSpacing()
        dimensions = image.GetSize()
        
        record = {
            'filename': filename,
            'patient_id': patient_id,
            'case_id': case_id,
            'class_label': class_label,
            'scan_idx': scan_idx,
            'modality': metadata.get('modality', 'nan'),
            'protocol': protocol,
            'scan_plane': scan_plane,
            'protocol_source': protocol_source,
            'image_orientation': metadata.get('image_orientation', 'nan'),
            'spacing_quality': metadata.get('spacing_quality', 'ok'),
            'nonuniformity_mm': metadata.get('nonuniformity_mm'),
            'sex': metadata.get('sex', 'nan'),
            'DOB': metadata.get('DOB', 'nan'),
            'age': metadata.get('age', 'nan'),
            'age_at_scan': metadata.get('age_at_scan', 'nan'),
            'DOS': metadata.get('DOS', 'nan'),
            'study_time': metadata.get('study_time', 'nan'),
            'series_description': metadata.get('series_description', 'nan'),
            'series_number': metadata.get('series_number', 'nan'),
            'manufacturer': metadata.get('manufacturer', 'nan'),
            'manufacturer_model': metadata.get('manufacturer_model', 'nan'),
            'institution': metadata.get('institution', 'nan'),
            'kvp': metadata.get('kvp', 'nan'),
            'contrast_agent': metadata.get('contrast_agent', 'nan'),
            'spacing_x': float(spacing[0]),
            'spacing_y': float(spacing[1]),
            'spacing_z': float(spacing[2]),
            'dim_x': int(dimensions[0]),
            'dim_y': int(dimensions[1]),
            'dim_z': int(dimensions[2]),
            'slice_thickness': metadata.get('slice_thickness', 'nan'),
            'num_slices': len(dicom_files),
            'conversion_date': datetime.now().isoformat(),
            'dataset_type': self.dataset_type,
            'output_path': str(output_path.relative_to(self.output_dir)),
            'validation_status': 'success' if validation['valid'] else 'warning',
        }
        
        self.conversion_records.append(record)
        self.class_distribution[class_label] += 1
        self.protocol_distribution[protocol] += 1
    
    def _generate_summary(self) -> Dict:
        unique_patients = set(r['patient_id'] for r in self.conversion_records)
        skipped_by_plane = defaultdict(int)
        skipped_by_reason = defaultdict(int)
        for skipped in self.skipped_series:
            plane = skipped.get('scan_plane', 'UNKNOWN')
            skipped_by_plane[plane] += 1
            
            if 'Too few slices' in skipped['reason']:
                skipped_by_reason['few_slices'] += 1
            elif skipped['stage'] == 'filtering':
                skipped_by_reason['non_axial'] += 1
            elif skipped['stage'] == 'quality_validation':
                skipped_by_reason['quality'] += 1
        
        return {
            'conversion_date': datetime.now().isoformat(),
            'dataset_type': self.dataset_type,
            'input_directory': str(self.input_dir),
            'output_directory': str(self.output_dir),
            'total_patients': len(unique_patients),
            'total_series_discovered': len(self.conversion_records) + len(self.skipped_series),
            'axial_series': len(self.conversion_records),
            'skipped_non_axial': len([s for s in self.skipped_series if s['stage'] == 'filtering' and 'Non-axial' in s['reason']]),
            'skipped_few_slices': len([s for s in self.skipped_series if 'Too few slices' in s['reason']]),
            'skipped_quality': len([s for s in self.skipped_series if s['stage'] == 'quality_validation']),
            'successful_conversions': len([r for r in self.conversion_records if r['validation_status'] == 'success']),
            'failed_conversions': len(self.failed_cases),
            'class_distribution': dict(self.class_distribution),
            'protocol_distribution': dict(self.protocol_distribution),
            'scan_plane_distribution': dict(self.scan_plane_distribution),
            'modality_distribution': dict(self.modality_distribution),
            'quality_distribution': dict(self.quality_issues),
            'patient_id_sources': dict(self.patient_id_sources),
            'skipped_breakdown': dict(skipped_by_plane),
            'failed_cases': self.failed_cases,
            'skipped_series': self.skipped_series[:100],
        }
    
    def _save_outputs(self, summary: Dict):
        summary_path = self.output_dir / 'conversion_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"      Summary saved to: {summary_path}")
        
        manifest_path = self.output_dir / 'manifest.csv'
        self._save_manifest_csv(manifest_path)
        print(f"      Manifest saved to: {manifest_path}")
    
    def _save_manifest_csv(self, manifest_path: Path):
        if not self.conversion_records:
            print(f"      No records to save")
            return
        
        columns = [
            'filename', 'patient_id', 'case_id', 'class_label', 'scan_idx', 
            'modality', 'protocol', 'scan_plane', 'protocol_source', 'image_orientation',
            'spacing_quality', 'nonuniformity_mm',
            'sex', 'DOB', 'age', 'age_at_scan',
            'DOS', 'study_time', 'series_description', 'series_number',
            'manufacturer', 'manufacturer_model', 'institution', 'kvp', 'contrast_agent',
            'spacing_x', 'spacing_y', 'spacing_z',
            'dim_x', 'dim_y', 'dim_z',
            'slice_thickness', 'num_slices',
            'conversion_date', 'dataset_type', 'output_path', 'validation_status'
        ]
        
        df = pd.DataFrame(self.conversion_records)
        existing_columns = [col for col in columns if col in df.columns]
        df = df[existing_columns]
        df.to_csv(manifest_path, index=False)
        print(f"      Saved {len(df)} records to manifest")
