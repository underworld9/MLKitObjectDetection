/*
 * Copyright 2020 Google LLC. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.mlkit.vision.demo;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.ImageFormat;
import android.hardware.Camera;
import android.hardware.Camera.CameraInfo;
import android.preference.PreferenceManager;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.WindowManager;

import com.google.android.gms.common.images.Size;

import java.io.IOException;
import java.lang.Thread.State;
import java.nio.ByteBuffer;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import androidx.annotation.Nullable;
import androidx.annotation.RequiresPermission;

/**
 * Manages the camera and allows UI updates on top of it (e.g. overlaying extra Graphics or
 * displaying extra information). This receives preview frames from the camera at a specified rate,
 * sending those frames to child classes' detectors / classifiers as fast as it is able to process.
 */
public class CameraSource {
    @SuppressLint("InlinedApi")

    public static final int IMAGE_FORMAT = ImageFormat.NV21;

    private static final String TAG = "MIDemoApp:CameraSource";

    protected Activity activity;

    private Camera camera;

    /**
     * Rotation of the device, and thus the associated preview images captured from the device.
     */
    private int rotationDegrees;

    private Size previewSize;

    private final GraphicOverlay graphicOverlay;

    // True if a SurfaceTexture is being used for the preview, false if a SurfaceHolder is being
    // used for the preview.  We want to be compatible back to Gingerbread, but SurfaceTexture
    // wasn't introduced until Honeycomb.  Since the interface cannot use a SurfaceTexture, if the
    // developer wants to display a preview we must use a SurfaceHolder.  If the developer doesn't
    // want to display a preview we use a SurfaceTexture if we are running at least Honeycomb.
    private boolean usingSurfaceTexture;

    /**
     * Dedicated thread and associated runnable for calling into the detector with frames, as the
     * frames become available from the camera.
     */
    private Thread processingThread;

    private final FrameProcessingRunnable processingRunnable;

    private final Object processorLock = new Object();
    // TODO(b/74400062) Re-enable the annotation
    // @GuardedBy("processorLock")
    private VisionImageProcessor frameProcessor;

    /**
     * Map to convert between a byte array, received from the camera, and its associated byte buffer.
     * We use byte buffers internally because this is a more efficient way to call into native code
     * later (avoids a potential copy).
     *
     * <p><b>Note:</b> uses IdentityHashMap here instead of HashMap because the behavior of an array's
     * equals, hashCode and toString methods is both useless and unexpected. IdentityHashMap enforces
     * identity ('==') check on the keys.
     */
    private final Map<byte[], ByteBuffer> bytesToByteBuffer = new IdentityHashMap<>();

    public CameraSource(Activity activity, GraphicOverlay overlay) {
        this.activity = activity;
        graphicOverlay = overlay;
        graphicOverlay.clear();
        processingRunnable = new FrameProcessingRunnable();
    }

    // ==============================================================================================
    // Public
    // ==============================================================================================

    /**
     * Stops the camera and releases the resources of the camera and underlying detector.
     */
    public void release() {
        synchronized (processorLock) {
            stop();
            processingRunnable.release();
            cleanScreen();

            if (frameProcessor != null) {
                frameProcessor.stop();
            }
        }
    }

    /**
     * Opens the camera and starts sending preview frames to the underlying detector. The supplied
     * surface holder is used for the preview so frames can be displayed to the user.
     *
     * @param surfaceHolder the surface holder to use for the preview frames
     * @throws IOException if the supplied surface holder could not be used as the preview display
     */
    @RequiresPermission(Manifest.permission.CAMERA)
    public synchronized CameraSource start(SurfaceHolder surfaceHolder) throws IOException {
        if (camera != null) {
            return this;
        }

        camera = createCamera();
        camera.setPreviewDisplay(surfaceHolder);
        camera.startPreview();

        processingThread = new Thread(processingRunnable);
        processingRunnable.setActive(true);
        processingThread.start();

        usingSurfaceTexture = false;
        return this;
    }

    /**
     * Gets the id for the camera specified by the direction it is facing. Returns -1 if no such
     * camera was found.
     */
    private static int getIdForRequestedCamera() {
        CameraInfo cameraInfo = new CameraInfo();
        for (int i = 0; i < Camera.getNumberOfCameras(); ++i) {
            Camera.getCameraInfo(i, cameraInfo);
            if (cameraInfo.facing == CameraInfo.CAMERA_FACING_BACK) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Returns the preview size that is currently in use by the underlying camera.
     */
    public Size getPreviewSize() {
        return previewSize;
    }

    /**
     * Closes the camera and stops sending frames to the underlying frame detector.
     *
     * <p>This camera source may be restarted again by calling {@link } or {@link
     * #start(SurfaceHolder)}.
     *
     * <p>Call {@link #release()} instead to completely shut down this camera source and release the
     * resources of the underlying detector.
     */
    public synchronized void stop() {
        processingRunnable.setActive(false);
        if (processingThread != null) {
            try {
                // Wait for the thread to complete to ensure that we can't have multiple threads
                // executing at the same time (i.e., which would happen if we called start too
                // quickly after stop).
                processingThread.join();
            } catch (InterruptedException e) {
                Log.d(TAG, "Frame processing thread interrupted on release.");
            }
            processingThread = null;
        }

        if (camera != null) {
            camera.stopPreview();
            camera.setPreviewCallbackWithBuffer(null);
            try {
                if (usingSurfaceTexture) {
                    camera.setPreviewTexture(null);
                } else {
                    camera.setPreviewDisplay(null);
                }
            } catch (Exception e) {
                Log.e(TAG, "Failed to clear camera preview: " + e);
            }
            camera.release();
            camera = null;
        }

        // Release the reference to any image buffers, since these will no longer be in use.
        bytesToByteBuffer.clear();
    }

    /**
     * Opens the camera and applies the user settings.
     */
    @SuppressLint("InlinedApi")
    private Camera createCamera() {
        int requestedCameraId = getIdForRequestedCamera();
        Camera camera = Camera.open(requestedCameraId);

        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(activity);
        SizePair sizePair = new SizePair(Size.parseSize((Objects.requireNonNull(sharedPreferences.getString(activity.getString(R.string.pref_key_rear_camera_preview_size), null)))),
                Size.parseSize((Objects.requireNonNull(sharedPreferences.getString(activity.getString(R.string.pref_key_rear_camera_picture_size), null)))));

        previewSize = sizePair.preview;

        float requestedFps = 30.0f;
        int[] previewFpsRange = selectPreviewFpsRange(camera, requestedFps);

        Camera.Parameters parameters = camera.getParameters();

        Size pictureSize = sizePair.picture;
        if (pictureSize != null) {
            parameters.setPictureSize(pictureSize.getWidth(), pictureSize.getHeight());
        }
        parameters.setPreviewSize(previewSize.getWidth(), previewSize.getHeight());
        parameters.setPreviewFpsRange(previewFpsRange[Camera.Parameters.PREVIEW_FPS_MIN_INDEX], previewFpsRange[Camera.Parameters.PREVIEW_FPS_MAX_INDEX]);
        // Use YV12 so that we can exercise YV12->NV21 auto-conversion logic for OCR detection
        parameters.setPreviewFormat(IMAGE_FORMAT);

        setRotation(camera, parameters, requestedCameraId);
        parameters.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO);

        camera.setParameters(parameters);

        // Four frame buffers are needed for working with the camera:
        //
        //   one for the frame that is currently being executed upon in doing detection
        //   one for the next pending frame to process immediately upon completing detection
        //   two for the frames that the camera uses to populate future preview images
        //
        // Through trial and error it appears that two free buffers, in addition to the two buffers
        // used in this code, are needed for the camera to work properly.  Perhaps the camera has
        // one thread for acquiring images, and another thread for calling into user code.  If only
        // three buffers are used, then the camera will spew thousands of warning messages when
        // detection takes a non-trivial amount of time.
        camera.setPreviewCallbackWithBuffer(new CameraPreviewCallback());
        camera.addCallbackBuffer(createPreviewBuffer(previewSize));
        camera.addCallbackBuffer(createPreviewBuffer(previewSize));
        camera.addCallbackBuffer(createPreviewBuffer(previewSize));
        camera.addCallbackBuffer(createPreviewBuffer(previewSize));

        return camera;
    }

    /**
     * Stores a preview size and a corresponding same-aspect-ratio picture size. To avoid distorted
     * preview images on some devices, the picture size must be set to a size that is the same aspect
     * ratio as the preview size or the preview may end up being distorted. If the picture size is
     * null, then there is no picture size with the same aspect ratio as the preview size.
     */
    public static class SizePair {
        public final Size preview;
        @Nullable
        public final Size picture;

        public SizePair(Size previewSize, @Nullable Size pictureSize) {
            preview = previewSize;
            picture = pictureSize;
        }
    }

    /**
     * Selects the most suitable preview frames per second range, given the desired frames per second.
     *
     * @param camera            the camera to select a frames per second range from
     * @param desiredPreviewFps the desired frames per second for the camera preview frames
     * @return the selected preview frames per second range
     */
    @SuppressLint("InlinedApi")
    private static int[] selectPreviewFpsRange(Camera camera, float desiredPreviewFps) {
        // The camera API uses integers scaled by a factor of 1000 instead of floating-point frame
        // rates.
        int desiredPreviewFpsScaled = (int) (desiredPreviewFps * 1000.0f);

        // The method for selecting the best range is to minimize the sum of the differences between
        // the desired value and the upper and lower bounds of the range.  This may select a range
        // that the desired value is outside of, but this is often preferred.  For example, if the
        // desired frame rate is 29.97, the range (30, 30) is probably more desirable than the
        // range (15, 30).
        int[] selectedFpsRange = null;
        int minDiff = Integer.MAX_VALUE;
        List<int[]> previewFpsRangeList = camera.getParameters().getSupportedPreviewFpsRange();
        for (int[] range : previewFpsRangeList) {
            int deltaMin = desiredPreviewFpsScaled - range[Camera.Parameters.PREVIEW_FPS_MIN_INDEX];
            int deltaMax = desiredPreviewFpsScaled - range[Camera.Parameters.PREVIEW_FPS_MAX_INDEX];
            int diff = Math.abs(deltaMin) + Math.abs(deltaMax);
            if (diff < minDiff) {
                selectedFpsRange = range;
                minDiff = diff;
            }
        }
        return selectedFpsRange;
    }

    /**
     * Calculates the correct rotation for the given camera id and sets the rotation in the
     * parameters. It also sets the camera's display orientation and rotation.
     *
     * @param parameters the camera parameters for which to set the rotation
     * @param cameraId   the camera id to set rotation based on
     */
    private void setRotation(Camera camera, Camera.Parameters parameters, int cameraId) {
        WindowManager windowManager = (WindowManager) activity.getSystemService(Context.WINDOW_SERVICE);
        int degrees = 0;
        int rotation;
        if (windowManager != null) {
            rotation = windowManager.getDefaultDisplay().getRotation();
            switch (rotation) {
                case Surface.ROTATION_0:
                    degrees = 0;
                    break;
                case Surface.ROTATION_90:
                    degrees = 90;
                    break;
                case Surface.ROTATION_180:
                    degrees = 180;
                    break;
                case Surface.ROTATION_270:
                    degrees = 270;
                    break;
                default:
                    Log.e(TAG, "Bad rotation value: " + rotation);
            }
        }
        CameraInfo cameraInfo = new CameraInfo();
        Camera.getCameraInfo(cameraId, cameraInfo);

        int displayAngle;
        this.rotationDegrees = (cameraInfo.orientation - degrees + 360) % 360;
        displayAngle = this.rotationDegrees;

        camera.setDisplayOrientation(displayAngle);
        parameters.setRotation(this.rotationDegrees);
    }

    /**
     * Creates one buffer for the camera preview callback. The size of the buffer is based off of the
     * camera preview size and the format of the camera image.
     *
     * @return a new preview buffer of the appropriate size for the current camera settings
     */
    @SuppressLint("InlinedApi")
    private byte[] createPreviewBuffer(Size previewSize) {
        int bitsPerPixel = ImageFormat.getBitsPerPixel(IMAGE_FORMAT);
        long sizeInBits = (long) previewSize.getHeight() * previewSize.getWidth() * bitsPerPixel;
        int bufferSize = (int) Math.ceil(sizeInBits / 8.0d) + 1;

        // Creating the byte array this way and wrapping it, as opposed to using .allocate(),
        // should guarantee that there will be an array to work with.
        byte[] byteArray = new byte[bufferSize];
        ByteBuffer buffer = ByteBuffer.wrap(byteArray);
        if (!buffer.hasArray() || (buffer.array() != byteArray)) {
            // I don't think that this will ever happen.  But if it does, then we wouldn't be
            // passing the preview content to the underlying detector later.
            throw new IllegalStateException("Failed to create valid buffer for camera source.");
        }

        bytesToByteBuffer.put(byteArray, buffer);
        return byteArray;
    }

    // ==============================================================================================
    // Frame processing
    // ==============================================================================================

    /**
     * Called when the camera has a new preview frame.
     */
    private class CameraPreviewCallback implements Camera.PreviewCallback {
        @Override
        public void onPreviewFrame(byte[] data, Camera camera) {
            processingRunnable.setNextFrame(data, camera);
        }
    }

    public void setMachineLearningFrameProcessor(VisionImageProcessor processor) {
        synchronized (processorLock) {
            cleanScreen();
            if (frameProcessor != null) {
                frameProcessor.stop();
            }
            frameProcessor = processor;
        }
    }

    /**
     * This runnable controls access to the underlying receiver, calling it to process frames when
     * available from the camera. This is designed to run detection on frames as fast as possible
     * (i.e., without unnecessary context switching or waiting on the next frame).
     *
     * <p>While detection is running on a frame, new frames may be received from the camera. As these
     * frames come in, the most recent frame is held onto as pending. As soon as detection and its
     * associated processing is done for the previous frame, detection on the mostly recently received
     * frame will immediately start on the same thread.
     */
    private class FrameProcessingRunnable implements Runnable {

        // This lock guards all of the member variables below.
        private final Object lock = new Object();
        private boolean active = true;

        // These pending variables hold the state associated with the new frame awaiting processing.
        private ByteBuffer pendingFrameData;

        FrameProcessingRunnable() {
        }

        /**
         * Releases the underlying receiver. This is only safe to do after the associated thread has
         * completed, which is managed in camera source's release method above.
         */
        @SuppressLint("Assert")
        void release() {
            assert (processingThread.getState() == State.TERMINATED);
        }

        /**
         * Marks the runnable as active/not active. Signals any blocked threads to continue.
         */
        void setActive(boolean active) {
            synchronized (lock) {
                this.active = active;
                lock.notifyAll();
            }
        }

        /**
         * Sets the frame data received from the camera. This adds the previous unused frame buffer (if
         * present) back to the camera, and keeps a pending reference to the frame data for future use.
         */
        @SuppressWarnings("ByteBufferBackingArray")
        void setNextFrame(byte[] data, Camera camera) {
            synchronized (lock) {
                if (pendingFrameData != null) {
                    camera.addCallbackBuffer(pendingFrameData.array());
                    pendingFrameData = null;
                }

                if (!bytesToByteBuffer.containsKey(data)) {
                    Log.d(
                            TAG,
                            "Skipping frame. Could not find ByteBuffer associated with the image "
                                    + "data from the camera.");
                    return;
                }

                pendingFrameData = bytesToByteBuffer.get(data);

                // Notify the processor thread if it is waiting on the next frame (see below).
                lock.notifyAll();
            }
        }

        /**
         * As long as the processing thread is active, this executes detection on frames continuously.
         * The next pending frame is either immediately available or hasn't been received yet. Once it
         * is available, we transfer the frame info to local variables and run detection on that frame.
         * It immediately loops back for the next frame without pausing.
         *
         * <p>If detection takes longer than the time in between new frames from the camera, this will
         * mean that this loop will run without ever waiting on a frame, avoiding any context switching
         * or frame acquisition time latency.
         *
         * <p>If you find that this is using more CPU than you'd like, you should probably decrease the
         * FPS setting above to allow for some idle time in between frames.
         */
        @SuppressLint("InlinedApi")
        @SuppressWarnings({"GuardedBy", "ByteBufferBackingArray"})
        @Override
        public void run() {
            ByteBuffer data;

            while (true) {
                synchronized (lock) {
                    while (active && (pendingFrameData == null)) {
                        try {
                            // Wait for the next frame to be received from the camera, since we
                            // don't have it yet.
                            lock.wait();
                        } catch (InterruptedException e) {
                            Log.d(TAG, "Frame processing loop terminated.", e);
                            return;
                        }
                    }

                    if (!active) {
                        // Exit the loop once this camera source is stopped or released.  We check
                        // this here, immediately after the wait() above, to handle the case where
                        // setActive(false) had been called, triggering the termination of this
                        // loop.
                        return;
                    }

                    // Hold onto the frame data locally, so that we can use this for detection
                    // below.  We need to clear pendingFrameData to ensure that this buffer isn't
                    // recycled back to the camera before we are done using that data.
                    data = pendingFrameData;
                    pendingFrameData = null;
                }

                // The code below needs to run outside of synchronization, because this will allow
                // the camera to add pending frame(s) while we are running detection on the current
                // frame.

                try {
                    synchronized (processorLock) {
                        frameProcessor.processByteBuffer(
                                data,
                                new FrameMetadata.Builder()
                                        .setWidth(previewSize.getWidth())
                                        .setHeight(previewSize.getHeight())
                                        .setRotation(rotationDegrees)
                                        .build(),
                                graphicOverlay);
                    }
                } catch (Exception t) {
                    Log.e(TAG, "Exception thrown from receiver.", t);
                } finally {
                    camera.addCallbackBuffer(data.array());
                }
            }
        }
    }

    /**
     * Cleans up graphicOverlay and child classes can do their cleanups as well .
     */
    private void cleanScreen() {
        graphicOverlay.clear();
    }
}
