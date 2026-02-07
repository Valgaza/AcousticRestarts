# Uploads Directory

This directory stores CSV files uploaded through the frontend interface for ML model processing.

## Structure

- Files are automatically saved with timestamps: `YYYYMMDD_HHMMSS_filename.csv`
- Example: `20260208_045457_traffic_data.csv`

## Purpose

CSV files uploaded here are intended to be processed by the ML model for traffic prediction.

## File Format

Only `.csv` files are accepted. Files must have the `.csv` extension.

## Cleanup

Consider periodically cleaning up old uploaded files to save disk space.
