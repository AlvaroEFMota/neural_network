use pixels::{Pixels, SurfaceTexture};
use winit::application::ApplicationHandler;
use winit::event_loop::ControlFlow;
use winit::event_loop::EventLoop;
use winit::event::WindowEvent;
use winit::event::ElementState;
use winit::event::MouseButton;
use winit::window::WindowId;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;
use winit::dpi::LogicalSize;
use std::sync::Arc;
use std::error::Error;
use std::time::Instant;
use std::f64::consts::PI;

#[derive(Default)]
struct Vec2 {
    x: f64,
    y: f64,
}

#[derive(Default)]
struct PhysicalObject {
    position: Vec2,
    velocity: Vec2,
    acceleration: Vec2,
}
struct App {
    window: Option<Arc<Window>>,
    pixels: Option<Pixels<'static>>,
    mouse_down: bool,
    meteor: PhysicalObject,
    planet: PhysicalObject,
    last_frame: Instant,
    cursor_position: Vec2,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            pixels: None,
            mouse_down: false,
            meteor: PhysicalObject {
                position: Vec2{x: 100.0, y: 100.0},
                velocity: Vec2{x: 1.0, y: 1.0},
                acceleration: Vec2{x: 0.0, y: 0.0},
            },
            planet: PhysicalObject {
                position: Vec2{x: 256.0, y: 256.0},
                velocity: Vec2{x: 0.0, y: 0.0},
                acceleration: Vec2{x: 0.0, y: 0.0},
            },
            last_frame: Instant::now(),
            cursor_position: Vec2{x: 0.0, y: 0.0},
        }
    }
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            pixels: None,
            mouse_down: false,
            meteor: PhysicalObject::default(),
            planet: PhysicalObject::default(),
            last_frame: Instant::now(),
            cursor_position: Vec2 { x: 0.0, y: 0.0 }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
       let window = Arc::new(
        event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("Draw")
                    .with_inner_size(LogicalSize::new(512.0, 512.0)),
            )
            .unwrap(),
        );

        let size = window.inner_size();

        let surface_texture =
            SurfaceTexture::new(size.width, size.height, window.clone());

        let pixels = Pixels::new(512, 512, surface_texture).unwrap();

        self.pixels = Some(pixels);
        self.window = Some(window); 
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            },

            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.mouse_down = state == ElementState::Pressed;
                    self.meteor.velocity.x += (self.cursor_position.x - self.meteor.position.x) / 10.0;
                    self.meteor.velocity.y += (self.cursor_position.y - self.meteor.position.y) / 10.0;
                }
            },

            WindowEvent::CursorMoved { position, .. } => {
                // update mouse position
                self.cursor_position.x = position.x;
                self.cursor_position.y = position.y;
                if self.mouse_down {
                    if let Some(pixels) = self.pixels.as_mut() {
                        let frame = pixels.frame_mut(); //[R, G, B, A, R, G, B, A, R, G, B, A, ...]

                        let x = position.x as usize;
                        let y = position.y as usize;

                        let index = (y * 512 + x) * 4;

                        if index + 3 < frame.len() {
                            frame[index] = 255; // R
                            frame[index + 1] = 255; // G
                            frame[index + 2] = 255; // B
                            frame[index + 3] = 255; // A
                        }
                    }
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
            }
            },

            WindowEvent::RedrawRequested => {

                let now = Instant::now();
                let delta = now.duration_since(self.last_frame).as_secs_f64();
                self.last_frame = now;

                //update
                
                if let Some(pixels) = self.pixels.as_mut() {
                    let frame = pixels.frame_mut(); //[R, G, B, A, R, G, B, A, R, G, B, A, ...]
                    frame.fill(0);

                    let planet_x = self.planet.position.x as usize;
                    let planet_y = self.planet.position.y as usize;

                    for i in 0..60 {
                        let angle_degree = (i * 6) as f64;
                        let angle_rad = angle_degree as f64 * (2.0 * PI / 360.0);
                        let mut planet_edge_x = (angle_rad.cos() * 20.0);
                        let mut planet_edge_y = (angle_rad.sin() * 20.0); 

                        planet_edge_x += planet_x as f64;
                        planet_edge_y += planet_y as f64;
                        let index = (planet_edge_y as usize * 512 + planet_edge_x as usize) * 4;
                        if index + 3 < frame.len() {
                            frame[index] = 255; // R
                            frame[index + 1] = 255; // G
                            frame[index + 2] = 255; // B
                            frame[index + 3] = 255; // A
                        }
                    }

                    // update meteor position
                    self.meteor.position.x += self.meteor.velocity.x * delta;
                    self.meteor.position.y += self.meteor.velocity.y * delta;
                    self.meteor.velocity.x += self.meteor.acceleration.x;
                    self.meteor.velocity.y += self.meteor.acceleration.y;
                    self.meteor.acceleration.x = (self.planet.position.x - self.meteor.position.x) / 100.0;
                    self.meteor.acceleration.y = (self.planet.position.y - self.meteor.position.y) / 100.0;

                    // Draw meteor
                    let meteor_x = self.meteor.position.x as usize;
                    let meteor_y = self.meteor.position.y as usize;

                    for i in 0..60 {
                        let angle_degree = (i * 6) as f64;
                        let angle_rad = angle_degree as f64 * (2.0 * PI / 360.0);
                        let mut meteor_edge_x = (angle_rad.cos() * 5.0);
                        let mut meteor_edge_y = (angle_rad.sin() * 5.0); 

                        meteor_edge_x += meteor_x as f64;
                        meteor_edge_y += meteor_y as f64;
                        let index = (meteor_edge_y as usize * 512 + meteor_edge_x as usize) * 4;
                        if index + 3 < frame.len() {
                            frame[index] = 255; // R
                            frame[index + 1] = 255; // G
                            frame[index + 2] = 255; // B
                            frame[index + 3] = 255; // A
                        }
                    }


                }

                if let Some(pixels) = self.pixels.as_mut() {
                    pixels.render().unwrap();
                }

                self.window.as_ref().unwrap().request_redraw();
            }
            _ => (),
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::new();
    let _ = event_loop.run_app(&mut app);
    Ok(())
}