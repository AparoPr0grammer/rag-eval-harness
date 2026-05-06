import "./globals.css";
import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";

export const metadata: Metadata = {
  title: "rag-eval dashboard",
  description: "Browse RAG evaluation run history",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header>
          <h1>rag-eval dashboard</h1>
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}
