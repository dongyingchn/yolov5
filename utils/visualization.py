# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Detection Box Visualization Utilities.

This module contains extracted and organized visualization code from detect.py for drawing
detection bounding boxes on images. It provides reusable functions for annotating, displaying,
and saving detection results.

Usage:
    from utils.visualization import annotate_detections, display_image, save_detection_results

Example:
    ```python
    from utils.visualization import annotate_detections, save_detection_results

    # Annotate detections on image
    annotated_image = annotate_detections(
        image=im0,
        detections=det,
        names=model.names,
        line_thickness=3,
        hide_labels=False,
        hide_conf=False
    )

    # Save annotated image
    save_detection_results(annotated_image, save_path)
    ```
"""

import platform
from pathlib import Path

import cv2
import torch
from ultralytics.utils.plotting import Annotator, colors, save_one_box

from utils.general import scale_boxes


def annotate_detections(
    image,
    detections,
    names,
    img_shape=None,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    save_crop=False,
    crop_save_dir=None,
    image_stem=None,
):
    """
    Annotate detection bounding boxes on an image.

    This function draws bounding boxes with labels and confidence scores on the input image
    based on detection results. It can also save cropped detection regions if requested.

    Args:
        image (numpy.ndarray): The input image in BGR format (HxWxC).
        detections (torch.Tensor): Detection results tensor with shape (N, 6) where each row
            contains [x1, y1, x2, y2, confidence, class_id].
        names (dict | list): Class names dictionary or list mapping class indices to names.
        img_shape (tuple, optional): Original inference image shape (H, W) for rescaling boxes.
            If None, boxes are assumed to be already in image coordinates.
        line_thickness (int): Bounding box line thickness in pixels. Default is 3.
        hide_labels (bool): If True, hide class labels on bounding boxes. Default is False.
        hide_conf (bool): If True, hide confidence scores on bounding boxes. Default is False.
        save_crop (bool): If True, save cropped detection regions. Default is False.
        crop_save_dir (Path | str, optional): Directory to save cropped images. Required if save_crop is True.
        image_stem (str, optional): Image file stem for naming cropped files. Required if save_crop is True.

    Returns:
        numpy.ndarray: The annotated image with detection boxes drawn.

    Example:
        ```python
        # Basic usage
        annotated = annotate_detections(im0, det, model.names)

        # With custom settings
        annotated = annotate_detections(
            image=im0,
            detections=det,
            names=model.names,
            img_shape=im.shape[2:],
            line_thickness=2,
            hide_conf=True
        )
        ```
    """
    # Make a copy for cropping if needed
    imc = image.copy() if save_crop else image

    # Create annotator
    annotator = Annotator(image, line_width=line_thickness, example=str(names))

    if len(detections):
        # Rescale boxes from inference size to image size if img_shape is provided
        if img_shape is not None:
            detections[:, :4] = scale_boxes(img_shape, detections[:, :4], image.shape).round()

        # Draw detection boxes
        for *xyxy, conf, cls in reversed(detections):
            c = int(cls)  # integer class
            if not hide_labels:
                label = names[c] if hide_conf else f"{names[c]} {conf:.2f}"
            else:
                label = None
            annotator.box_label(xyxy, label, color=colors(c, True))

            # Save cropped detection if requested
            if save_crop and crop_save_dir is not None and image_stem is not None:
                save_one_box(
                    xyxy,
                    imc,
                    file=Path(crop_save_dir) / "crops" / names[c] / f"{image_stem}.jpg",
                    BGR=True,
                )

    return annotator.result()


def display_image(image, window_name, windows=None):
    """
    Display an image in a window using OpenCV.

    This function creates a window and displays the image. On Linux systems, it creates
    a resizable window that matches the image dimensions.

    Args:
        image (numpy.ndarray): The image to display in BGR format.
        window_name (str | Path): Name/title of the display window.
        windows (list, optional): List to track created windows. If provided and the window
            is new, it will be appended to this list.

    Returns:
        list | None: Updated windows list if provided, None otherwise.

    Example:
        ```python
        windows = []
        for annotated_image in annotated_images:
            windows = display_image(annotated_image, "Detection Results", windows)
        ```
    """
    window_name = str(window_name)

    # Track windows and create resizable window on Linux
    if platform.system() == "Linux":
        if windows is not None and window_name not in windows:
            windows.append(window_name)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(window_name, image.shape[1], image.shape[0])

    cv2.imshow(window_name, image)
    cv2.waitKey(1)  # 1 millisecond

    return windows


def save_detection_results(
    image,
    save_path,
    mode="image",
    vid_path=None,
    vid_writer=None,
    vid_cap=None,
):
    """
    Save detection results to an image or video file.

    This function handles saving annotated images or video frames. For videos, it manages
    the video writer lifecycle, creating a new writer when the output path changes.

    Args:
        image (numpy.ndarray): The annotated image to save in BGR format.
        save_path (str | Path): Path to save the output file.
        mode (str): Output mode - "image" for single images, "video" or "stream" for video.
            Default is "image".
        vid_path (str, optional): Current video path being written. Used to detect video changes.
        vid_writer (cv2.VideoWriter, optional): Existing video writer object.
        vid_cap (cv2.VideoCapture, optional): Video capture object to get video properties.

    Returns:
        tuple: (vid_path, vid_writer) - Updated video path and writer for video mode.
               (None, None) for image mode.

    Example:
        ```python
        # Save single image
        save_detection_results(annotated_image, "output/result.jpg")

        # Save video frames
        vid_path, vid_writer = None, None
        for frame in frames:
            vid_path, vid_writer = save_detection_results(
                frame,
                "output/result.mp4",
                mode="video",
                vid_path=vid_path,
                vid_writer=vid_writer,
                vid_cap=cap
            )
        ```
    """
    save_path = str(save_path)

    if mode == "image":
        cv2.imwrite(save_path, image)
        return None, None
    else:  # 'video' or 'stream'
        if vid_path != save_path:  # new video
            vid_path = save_path
            if isinstance(vid_writer, cv2.VideoWriter):
                vid_writer.release()  # release previous video writer
            if vid_cap:  # video
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:  # stream
                fps, w, h = 30, image.shape[1], image.shape[0]
            save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        vid_writer.write(image)
        return vid_path, vid_writer


def get_detection_info(detections, names):
    """
    Extract detection information from detection tensor.

    This function parses the detection tensor and returns structured information about
    each detection including class name, confidence, and bounding box coordinates.

    Args:
        detections (torch.Tensor): Detection results tensor with shape (N, 6) where each row
            contains [x1, y1, x2, y2, confidence, class_id].
        names (dict | list): Class names dictionary or list mapping class indices to names.

    Returns:
        list[dict]: List of detection dictionaries, each containing:
            - "class_id" (int): Class index
            - "class_name" (str): Class name
            - "confidence" (float): Detection confidence score
            - "bbox" (list): Bounding box coordinates [x1, y1, x2, y2]

    Example:
        ```python
        info = get_detection_info(det, model.names)
        for d in info:
            print(f"Detected {d['class_name']} with {d['confidence']:.2f} confidence")
        ```
    """
    results = []
    for *xyxy, conf, cls in detections:
        c = int(cls)
        results.append(
            {
                "class_id": c,
                "class_name": names[c],
                "confidence": float(conf),
                "bbox": [float(x) for x in xyxy],
            }
        )
    return results


def count_detections_by_class(detections, names):
    """
    Count the number of detections for each class.

    Args:
        detections (torch.Tensor): Detection results tensor with shape (N, 6) where each row
            contains [x1, y1, x2, y2, confidence, class_id].
        names (dict | list): Class names dictionary or list mapping class indices to names.

    Returns:
        dict: Dictionary mapping class names to detection counts.

    Example:
        ```python
        counts = count_detections_by_class(det, model.names)
        for class_name, count in counts.items():
            print(f"{count} {class_name}{'s' if count > 1 else ''}")
        ```
    """
    counts = {}
    if len(detections):
        for c in detections[:, 5].unique():
            c = int(c)
            n = int((detections[:, 5] == c).sum())
            class_name = names[c]
            counts[class_name] = n
    return counts


def format_detection_string(detections, names, image_shape=None):
    """
    Format detection results into a human-readable string.

    Args:
        detections (torch.Tensor): Detection results tensor with shape (N, 6).
        names (dict | list): Class names dictionary or list.
        image_shape (tuple, optional): Image shape (H, W) to include in the string.

    Returns:
        str: Formatted detection string, e.g., "640x480 2 persons, 1 car".

    Example:
        ```python
        s = format_detection_string(det, model.names, im.shape[2:])
        print(s)  # "640x480 2 persons, 1 car"
        ```
    """
    result = ""

    # Add image shape if provided
    if image_shape is not None:
        result = f"{image_shape[0]:g}x{image_shape[1]:g} "

    # Add detection counts
    if len(detections):
        detection_parts = []
        for c in detections[:, 5].unique():
            n = int((detections[:, 5] == c).sum())
            class_name = names[int(c)]
            detection_parts.append(f"{n} {class_name}{'s' * (n > 1)}")
        result += ", ".join(detection_parts)

    return result
