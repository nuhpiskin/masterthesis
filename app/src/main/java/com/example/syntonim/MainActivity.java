package com.example.syntonim;

import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.PixelFormat;
import android.os.Bundle;
import android.os.Environment;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.Spinner;
import android.widget.Switch;
import android.widget.TextView;
import android.view.SurfaceView;
import android.view.Surface;

import android.content.res.AssetManager;

import android.util.Log;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.File;
import java.io.IOException;

public class MainActivity extends Activity implements SurfaceHolder.Callback{
    public static final int REQUEST_CAMERA = 100;

    private int facing = 0;
    Bitmap bitmapIn = null;
    String target_name = "elon.jpg";
    private Spinner spinnerModel;
    private Spinner spinnerCPUGPU;
    private int current_face = 0;
    private int current_cpugpu = 0;
    private SurfaceView cameraView;



    @Override
    public void onCreate(Bundle savedInstanceState)
    {



        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);





    }
}
