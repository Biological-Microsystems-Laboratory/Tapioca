from pathlib import Path

import cv2

from Tapioca.segment_hardcode import image_segmenter

FIN_RES = Path("../Results (mSAM)")
FIN_RES.mkdir(exist_ok=True)
segmenter = image_segmenter(
        WEIGHTS=Path("Tapioca/mobile_sam.pt"),
        RESULTS=FIN_RES,
        MODEL="mSAM",
        DEBUG=True,
        SAVE_RESULTS=True,
        COCO=False,
        SCALE=6.0755
    )

# Load a test image
test_image_path = Path("../DropletPNGS/60_30_1.png")

if not test_image_path.exists():
    print(f"Test image not found: {test_image_path}")
else:
    # Process the test image
    result_image = segmenter.gen_seg(test_image_path)

    # Save the result image
    cv2.imshow("Result", result_image)
    cv2.imwrite(str(FIN_RES / "FIN.jpg"), result_image)
    # print(f"Result image saved to: {result_image_path}")
    cv2.waitKey(0)

