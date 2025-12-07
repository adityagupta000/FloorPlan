"use client";

import { Suspense, useRef, useState, useEffect } from "react";
import { Canvas, useFrame, useLoader } from "@react-three/fiber";
import { OrbitControls, Environment, Grid, Html } from "@react-three/drei";
import { OBJLoader } from "three-stdlib";
import * as THREE from "three";
import { Box, RotateCw, Eye, Maximize2, Minimize2, X } from "lucide-react";
import { Button } from "@/components/ui/button";

interface Model3DViewerProps {
  objUrl: string | null;
}

// Component to load and display the actual OBJ model with colors
function LoadedOBJModel({ url }: { url: string }) {
  const groupRef = useRef<THREE.Group>(null);
  const obj = useLoader(OBJLoader, url);

  useEffect(() => {
    if (obj) {
      // Center the model
      const box = new THREE.Box3().setFromObject(obj);
      const center = box.getCenter(new THREE.Vector3());
      obj.position.sub(center);

      // Scale to fit viewport
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      const scale = 3 / maxDim;
      obj.scale.setScalar(scale);

      // Apply colors to meshes based on vertex colors or create default colors
      obj.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          // Check if the mesh has vertex colors
          if (child.geometry.attributes.color) {
            // Use vertex colors
            child.material = new THREE.MeshStandardMaterial({
              vertexColors: true,
              metalness: 0.1,
              roughness: 0.7,
            });
          } else {
            // If no vertex colors, apply a default material
            child.material = new THREE.MeshStandardMaterial({
              color: 0x888888,
              metalness: 0.1,
              roughness: 0.7,
            });
          }
        }
      });
    }
  }, [obj]);

  return <primitive ref={groupRef} object={obj} />;
}

// Placeholder model with colors (fallback)
function PlaceholderModel() {
  const groupRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = state.clock.elapsedTime * 0.2;
    }
  });

  return (
    <group ref={groupRef}>
      {/* Floor - Light Yellow/Beige */}
      <mesh position={[0, -0.05, 0]} receiveShadow>
        <boxGeometry args={[3, 0.1, 3]} />
        <meshStandardMaterial color="#f5e6d3" metalness={0.1} roughness={0.8} />
      </mesh>

      {/* Outer Walls - Gray */}
      <mesh position={[0, 0.4, -1.45]}>
        <boxGeometry args={[3, 0.8, 0.1]} />
        <meshStandardMaterial color="#888888" metalness={0.1} roughness={0.7} />
      </mesh>
      <mesh position={[0, 0.4, 1.45]}>
        <boxGeometry args={[3, 0.8, 0.1]} />
        <meshStandardMaterial color="#888888" metalness={0.1} roughness={0.7} />
      </mesh>
      <mesh position={[-1.45, 0.4, 0]}>
        <boxGeometry args={[0.1, 0.8, 3]} />
        <meshStandardMaterial color="#888888" metalness={0.1} roughness={0.7} />
      </mesh>
      <mesh position={[1.45, 0.4, 0]}>
        <boxGeometry args={[0.1, 0.8, 3]} />
        <meshStandardMaterial color="#888888" metalness={0.1} roughness={0.7} />
      </mesh>

      {/* Inner Walls - Gray */}
      <mesh position={[0, 0.4, 0]}>
        <boxGeometry args={[0.08, 0.8, 1.5]} />
        <meshStandardMaterial color="#888888" metalness={0.1} roughness={0.7} />
      </mesh>
      <mesh position={[-0.7, 0.4, 0.7]}>
        <boxGeometry args={[1.4, 0.8, 0.08]} />
        <meshStandardMaterial color="#888888" metalness={0.1} roughness={0.7} />
      </mesh>

      {/* Door - Brown */}
      <mesh position={[0.5, 0.25, 0.7]}>
        <boxGeometry args={[0.4, 0.5, 0.06]} />
        <meshStandardMaterial color="#8B4513" metalness={0.2} roughness={0.6} />
      </mesh>

      {/* Windows - Light Blue with transparency */}
      <mesh position={[-1.44, 0.5, -0.5]}>
        <boxGeometry args={[0.08, 0.3, 0.5]} />
        <meshStandardMaterial
          color="#87CEEB"
          transparent
          opacity={0.6}
          metalness={0.3}
          roughness={0.3}
        />
      </mesh>
      <mesh position={[1.44, 0.5, 0.5]}>
        <boxGeometry args={[0.08, 0.3, 0.5]} />
        <meshStandardMaterial
          color="#87CEEB"
          transparent
          opacity={0.6}
          metalness={0.3}
          roughness={0.3}
        />
      </mesh>
    </group>
  );
}

function LoadingFallback() {
  return (
    <Html center>
      <div className="flex flex-col items-center gap-3 text-muted-foreground">
        <RotateCw className="w-8 h-8 animate-spin" />
        <span className="text-sm">Loading 3D Model...</span>
      </div>
    </Html>
  );
}

function EmptyState() {
  return (
    <div className="w-full h-full flex flex-col items-center justify-center text-muted-foreground">
      <Box className="w-16 h-16 mb-4 opacity-50" />
      <p className="text-sm">3D Preview will appear here</p>
    </div>
  );
}

export function Model3DViewer({ objUrl }: Model3DViewerProps) {
  const [isClient, setIsClient] = useState(false);
  const [loadError, setLoadError] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setIsClient(true);
  }, []);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => {
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
    };
  }, []);

  const toggleFullscreen = async () => {
    if (!containerRef.current) return;

    try {
      if (!document.fullscreenElement) {
        await containerRef.current.requestFullscreen();
      } else {
        await document.exitFullscreen();
      }
    } catch (error) {
      console.error("Error toggling fullscreen:", error);
    }
  };

  if (!objUrl) {
    return <EmptyState />;
  }

  if (!isClient) {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <RotateCw className="w-8 h-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div ref={containerRef} className="relative w-full h-full bg-background">
      <Canvas
        camera={{ position: [4, 4, 4], fov: 50 }}
        style={{ background: "transparent" }}
      >
        <ambientLight intensity={0.6} />
        <directionalLight position={[10, 10, 5]} intensity={1} castShadow />
        <directionalLight position={[-10, -10, -5]} intensity={0.3} />
        <pointLight position={[0, 5, 0]} intensity={0.5} />

        <Suspense fallback={<LoadingFallback />}>
          {!loadError ? <LoadedOBJModel url={objUrl} /> : <PlaceholderModel />}
        </Suspense>

        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          minDistance={2}
          maxDistance={15}
          autoRotate
          autoRotateSpeed={0.1}
        />

        <Grid
          infiniteGrid
          fadeDistance={15}
          fadeStrength={1}
          cellSize={0.5}
          cellThickness={0.5}
          cellColor="#444444"
          sectionSize={2}
          sectionThickness={1}
          sectionColor="#666666"
        />

        <Environment preset="city" />
      </Canvas>

      {/* Fullscreen Toggle Button */}
      <div className="absolute top-3 left-3 z-10">
        <Button
          variant="secondary"
          size="icon"
          onClick={toggleFullscreen}
          className="bg-background/80 backdrop-blur-sm hover:bg-background/90 shadow-lg"
          title={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
        >
          {isFullscreen ? (
            <Minimize2 className="w-4 h-4" />
          ) : (
            <Maximize2 className="w-4 h-4" />
          )}
        </Button>
      </div>

      {/* Overlay info */}
      <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs text-muted-foreground bg-background/80 backdrop-blur-sm px-3 py-2 rounded-lg">
          <Eye className="w-3 h-3" />
          <span>Drag to rotate | Scroll to zoom</span>
        </div>
      </div>

      {/* Color Legend */}
      <div className="absolute top-3 right-3 bg-background/90 backdrop-blur-sm p-3 rounded-lg border border-border shadow-lg">
        <p className="text-xs font-semibold mb-2">Colors:</p>
        <div className="space-y-1">
          {/* Walls */}
          <div className="flex items-center gap-2 text-xs">
            <div
              className="w-3 h-3 rounded-sm"
              style={{ backgroundColor: "#B4B4B4" }}
            />
            <span>Walls</span>
          </div>

          {/* Doors */}
          <div className="flex items-center gap-2 text-xs">
            <div
              className="w-3 h-3 rounded-sm"
              style={{ backgroundColor: "#A5372D" }}
            />
            <span>Doors</span>
          </div>

          {/* Window Frames */}
          <div className="flex items-center gap-2 text-xs">
            <div
              className="w-3 h-3 rounded-sm"
              style={{ backgroundColor: "#FFB41E" }}
            />
            <span>Window Frames</span>
          </div>

          {/* Glass */}
          <div className="flex items-center gap-2 text-xs">
            <div
              className="w-3 h-3 rounded-sm"
              style={{
                backgroundColor: "#32AAFF",
                opacity: 0.6,
              }}
            />
            <span>Glass</span>
          </div>

          {/* Floor */}
          <div className="flex items-center gap-2 text-xs">
            <div
              className="w-3 h-3 rounded-sm"
              style={{ backgroundColor: "#D2BE96" }}
            />
            <span>Floor</span>
          </div>
        </div>
      </div>
    </div>
  );
}
