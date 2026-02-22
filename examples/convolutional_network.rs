use pixels::{Pixels, SurfaceTexture};
use std::error::Error;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};
use std::sync::Arc;

const SCALE: u32 = 10;

struct App {
    window: Option<Arc<Window>>,
    pixels: Option<Pixels<'static>>,
    image_data: Vec<u8>,
    width: u32,
    height: u32,
}

impl App {
    fn new(image_path: &str) -> Result<Self, Box<dyn Error>> {
        // Load image and convert to RGBA8
        let img = image::open(image_path)?.to_rgba8();
        let (width, height) = img.dimensions();

        Ok(Self {
            window: None,
            pixels: None,
            image_data: img.to_vec(),
            width,
            height,
        })
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Create window
        let window = Arc::new(
            event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("Image Viewer")
                    .with_inner_size(PhysicalSize::new(SCALE * self.width, SCALE * self.height)),
            )
            .unwrap(),
        );

        // Create pixel surface
        let surface_texture = SurfaceTexture::new(SCALE * self.width, SCALE * self.height, window.clone());
        let mut pixels =
            Pixels::new(self.width, self.height, surface_texture).unwrap();
        //Duplicated image logic
        //let mut pixels =
        //    Pixels::new(2 * self.width, self.height, surface_texture).unwrap();

        // Copy image into frame buffer

        //Duplicated image logic
        /*let frame = pixels.frame_mut();
        let frame_width = 2 * self.width;

        for y in 0..self.height {
            // image row source and end
            let img_src = (y * self.width * 4) as usize;
            let img_end = img_src + (self.width * 4) as usize;

            //frame row source and end
            let frame_src = (y * frame_width * 4) as usize;

            //left image
            frame[frame_src..frame_src + (self.width * 4) as usize].copy_from_slice(&self.image_data[img_src..img_end]);
        
            //right image
            let right_start = frame_src + (self.width * 4) as usize;
            frame[right_start..right_start + (self.width * 4) as usize].copy_from_slice(&self.image_data[img_src..img_end]);
        }*/
        pixels.frame_mut().copy_from_slice(&self.image_data);

        self.pixels = Some(pixels);
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::RedrawRequested => {
                if let Some(pixels) = &mut self.pixels {
                    pixels.render().unwrap();
                }
            }

            WindowEvent::Resized(size) => {
                if let Some(pixels) = &mut self.pixels {
                    pixels.resize_surface(size.width, size.height).unwrap();
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {


    //let img = ImageReader::open("C:\\Users\\Shaka\\Downloads\\archive\\mnist_png\\test\\4\\4.png")?.decode()?;
    //println!("{:?}", img.dimensions());
    //let (img_w, img_h) = img.dimensions();

    let event_loop = EventLoop::new()?;
    let mut app = App::new("C:\\Users\\Shaka\\Downloads\\archive\\mnist_png\\test\\4\\4.png")?;
    event_loop.run_app(&mut app)?;

    Ok(())

    /*for i in 0..27 {
        for j in 0..27 {
            let pixel = img.get_pixel(i, j);
            println!("{:?} ", pixel); 
        }
    }*/

}