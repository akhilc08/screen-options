"""On-device text recognition with Apple Vision.

Returns text lines with boxes in the window's point space (top-left origin), so
they can be matched against AX frames.
"""
import Vision


def ocr_lines(cgimage, frame, fast=False, min_conf=0.3):
    """-> list of {"text", "box": (x, y, w, h) in screen points, "conf"} for one window capture."""
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelFast if fast else Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(not fast)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cgimage, None)
    ok, err = handler.performRequests_error_([req], None)
    if not ok:
        raise RuntimeError(f"Vision OCR failed: {err}")
    fx, fy, fw, fh = frame
    out = []
    for obs in req.results() or []:
        cand = obs.topCandidates_(1)
        if not cand or cand[0].confidence() < min_conf:
            continue
        bb = obs.boundingBox()        # normalized, bottom-left origin
        x, y = bb.origin.x, bb.origin.y
        w, h = bb.size.width, bb.size.height
        out.append({
            "text": cand[0].string(),
            "box": (round(fx + x * fw), round(fy + (1 - y - h) * fh), round(w * fw), round(h * fh)),
            "conf": round(float(cand[0].confidence()), 3),
        })
    return out
