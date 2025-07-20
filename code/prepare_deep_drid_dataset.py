import os
import pandas as pd
import shutil
from pathlib import Path

def prepare_deep_drid_dataset():
    """
    Prepare DeepDRiD dataset by organizing images according to patient DR levels.
    Images are classified from 0-4 based on patient_DR_Level, but only if the
    specific eye's DR level matches the patient's DR level.
    """

    # Define paths
    base_path = Path(r"c:") # This should be the base path the dataset is located
    csv_path = base_path / "Other_dataset_for_validation" / "DeepDRiD" / "regular_fundus_images" / "regular-fundus-validation" / "regular-fundus-validation.csv"
    images_path = base_path / "Other_dataset_for_validation" / "DeepDRiD" / "regular_fundus_images" / "regular-fundus-validation" / "Images"

    # Updated output path to match your enhanced inference script
    output_path = base_path / "2015" / "multi_class_classification" / "data" / "DeepDRID"

    # Define class names as requested
    class_names = [
        "0 - No DR",
        "1 - Mild",
        "2 - Moderate",
        "3 - Severe",
        "4 - Proliferative DR"
    ]

    # Create output directories for each class (0-4) with descriptive names
    for class_label, class_name in enumerate(class_names):
        class_dir = output_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {class_dir}")

    # Read the CSV file
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)

    # Print dataset info
    print(f"Total records in CSV: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    # Process each row
    processed_count = 0
    skipped_count = 0
    mismatch_count = 0

    for idx, row in df.iterrows():
        image_id = row['image_id']
        patient_dr_level = row['patient_DR_Level']
        left_dr_level = row['left_eye_DR_Level']
        right_dr_level = row['right_eye_DR_Level']
        patient_id = row['patient_id']

        # Use patient_DR_Level as the main classification criterion
        if pd.notna(patient_dr_level):
            dr_level = int(patient_dr_level)
        else:
            print(f"Warning: No patient DR level found for image {image_id}")
            skipped_count += 1
            continue

        # Skip if DR level is out of range (0-4)
        if dr_level < 0 or dr_level > 4:
            print(f"Warning: DR level {dr_level} out of range for image {image_id}")
            skipped_count += 1
            continue

        # Determine eye type and check if eye-specific DR level matches patient DR level
        eye_type = None
        eye_dr_matches = False

        if pd.notna(left_dr_level):
            eye_type = "left"
            if int(left_dr_level) == dr_level:
                eye_dr_matches = True
            else:
                print(f"Mismatch for {image_id}: left eye DR level {int(left_dr_level)} != patient DR level {dr_level}")

        elif pd.notna(right_dr_level):
            eye_type = "right"
            if int(right_dr_level) == dr_level:
                eye_dr_matches = True
            else:
                print(f"Mismatch for {image_id}: right eye DR level {int(right_dr_level)} != patient DR level {dr_level}")
        else:
            print(f"Warning: No eye-specific DR level found for image {image_id}")
            skipped_count += 1
            continue

        # Only process images where eye DR level matches patient DR level
        if not eye_dr_matches:
            mismatch_count += 1
            continue

        # Construct source image path
        source_image = images_path / str(patient_id) / f"{image_id}.jpg"

        # Check if source image exists
        if not source_image.exists():
            print(f"Warning: Image not found: {source_image}")
            skipped_count += 1
            continue

        # Construct destination path using descriptive class name
        dest_dir = output_path / class_names[dr_level]
        dest_image = dest_dir / f"{image_id}_dr{dr_level}_{eye_type}.jpg"

        # Copy image to destination
        try:
            shutil.copy2(source_image, dest_image)
            processed_count += 1

            if processed_count % 50 == 0:
                print(f"Processed {processed_count} images...")

        except Exception as e:
            print(f"Error copying {source_image} to {dest_image}: {e}")
            skipped_count += 1

    # Print summary
    print("\n" + "="*50)
    print("DATASET PREPARATION SUMMARY")
    print("="*50)
    print(f"Total images processed: {processed_count}")
    print(f"Total images skipped (missing files/invalid data): {skipped_count}")
    print(f"Total images skipped (DR level mismatch): {mismatch_count}")
    print(f"Total images in CSV: {len(df)}")

    # Print class distribution
    print("\nClass distribution:")
    for class_label, class_name in enumerate(class_names):
        class_dir = output_path / class_name
        count = len(list(class_dir.glob("*.jpg")))
        print(f"{class_name}: {count} images")

    print(f"\nPrepared dataset saved to: {output_path}")
    return output_path

def analyze_csv_data():
    """
    Analyze the CSV data to understand the distribution of DR levels and mismatches.
    """
    base_path = Path(r"c:\Users\BAU LAB\dr_ruth_alabi\DR_15_19")
    csv_path = base_path / "Other_dataset_for_validation" / "DeepDRiD" / "regular_fundus_images" / "regular-fundus-validation" / "regular-fundus-validation.csv"

    df = pd.read_csv(csv_path)

    print("CSV Data Analysis")
    print("="*30)
    print(f"Total records: {len(df)}")

    # Analyze patient DR levels (this is what we'll use)
    patient_dr_counts = df['patient_DR_Level'].value_counts().sort_index()
    print("\nPatient DR level distribution:")
    print(patient_dr_counts)

    # Show eye-specific levels for reference
    left_eye_counts = df['left_eye_DR_Level'].value_counts().sort_index()
    print("\nLeft eye DR level distribution:")
    print(left_eye_counts)

    right_eye_counts = df['right_eye_DR_Level'].value_counts().sort_index()
    print("\nRight eye DR level distribution:")
    print(right_eye_counts)

    # Analyze matching criteria
    print("\n" + "="*50)
    print("MATCHING ANALYSIS (Eye DR Level vs Patient DR Level)")
    print("="*50)

    class_names = [
        "0 - No DR",
        "1 - Mild",
        "2 - Moderate",
        "3 - Severe",
        "4 - Proliferative DR"
    ]

    total_matching = 0
    total_mismatched = 0

    for level in range(5):
        # Count images where patient DR level = level AND eye DR level also = level
        left_matching = len(df[(df['patient_DR_Level'] == level) & (df['left_eye_DR_Level'] == level)])
        right_matching = len(df[(df['patient_DR_Level'] == level) & (df['right_eye_DR_Level'] == level)])
        total_level_matching = left_matching + right_matching

        # Count images where patient DR level = level BUT eye DR level != level
        left_mismatched = len(df[(df['patient_DR_Level'] == level) &
                                (df['left_eye_DR_Level'].notna()) &
                                (df['left_eye_DR_Level'] != level)])
        right_mismatched = len(df[(df['patient_DR_Level'] == level) &
                                 (df['right_eye_DR_Level'].notna()) &
                                 (df['right_eye_DR_Level'] != level)])
        total_level_mismatched = left_mismatched + right_mismatched

        print(f"{class_names[level]}:")
        print(f"  Matching images: {total_level_matching} (L:{left_matching}, R:{right_matching})")
        print(f"  Mismatched images: {total_level_mismatched} (L:{left_mismatched}, R:{right_mismatched})")

        total_matching += total_level_matching
        total_mismatched += total_level_mismatched

    print(f"\nOVERALL SUMMARY:")
    print(f"Total matching images (will be included): {total_matching}")
    print(f"Total mismatched images (will be excluded): {total_mismatched}")
    print(f"Matching rate: {total_matching / (total_matching + total_mismatched) * 100:.1f}%")

    # Show eye type distribution for matching images
    left_eye_images = len(df[(df['left_eye_DR_Level'].notna()) &
                            (df['left_eye_DR_Level'] == df['patient_DR_Level'])])
    right_eye_images = len(df[(df['right_eye_DR_Level'].notna()) &
                             (df['right_eye_DR_Level'] == df['patient_DR_Level'])])

    print(f"\nEye type distribution (matching images only):")
    print(f"Left eye images: {left_eye_images}")
    print(f"Right eye images: {right_eye_images}")
    print(f"Total matching images: {left_eye_images + right_eye_images}")

def clean_existing_dataset():
    """
    Clean up any existing dataset preparation to start fresh.
    """
    base_path = Path(r"c:") # This should be the base path the dataset is located
    output_path = base_path / "2015" / "multi_class_classification" / "data" / "DeepDRID"

    if output_path.exists():
        print(f"Cleaning existing dataset at: {output_path}")
        shutil.rmtree(output_path)
        print("✅ Existing dataset cleaned!")
    else:
        print("No existing dataset found to clean.")

if __name__ == "__main__":
    print("DeepDRiD Dataset Preparation Script (Matching Eye & Patient DR Levels)")
    print("="*70)

    # First analyze the data
    analyze_csv_data()

    print("\n" + "="*70)
    print("This will prepare the dataset using the following criteria:")
    print("1. Use patient_DR_Level as the main classification")
    print("2. Only include images where eye DR level MATCHES patient DR level")
    print("3. Images will be organized into descriptive class folders:")

    class_names = [
        "0 - No DR",
        "1 - Mild",
        "2 - Moderate",
        "3 - Severe",
        "4 - Proliferative DR"
    ]

    for class_name in class_names:
        print(f"  - {class_name}")

    print("\nThis ensures data quality by only using images where the specific")
    print("eye's DR level aligns with the overall patient classification.")

    response = input("\nProceed with dataset preparation? (y/n): ")

    if response.lower() in ['y', 'yes']:
        # Clean existing dataset first
        clean_existing_dataset()

        # Prepare the new dataset
        output_dir = prepare_deep_drid_dataset()

        print(f"\n🎉 Dataset preparation completed!")
        print(f"📁 Output directory: {output_dir}")
        print(f"🔗 This matches the path expected by your enhanced inference script!")

        # Verify the preparation
        print(f"\n📊 Final verification:")
        for class_name in class_names:
            class_dir = output_dir / class_name
            if class_dir.exists():
                count = len(list(class_dir.glob("*.jpg")))
                print(f"✅ {class_name}: {count} images")
            else:
                print(f"❌ {class_name}: Directory not found!")

    else:
        print("Dataset preparation cancelled.")
