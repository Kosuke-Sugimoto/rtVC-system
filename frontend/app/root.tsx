import {
  Links,
  Scripts
} from "@remix-run/react";
import type { LinksFunction } from "@remix-run/node";

import rootStyle from "./styles/root.css?url";
import choiceButtonStyle from "./styles/choice-button.css?url";

import { ChoiceButtons } from "./components/ChoiceButtons";

export const links: LinksFunction = () => [
  { rel: "stylesheet", href: rootStyle },
  { rel: "stylesheet", href: choiceButtonStyle }
];

export default function App() {
  return (
    <html lang="ja">
      <head>
        <meta charSet="utf-8" />
        <meta
          name="viewport"
          content="width=device-width, initial-scale=1"
        />
        <Links />
      </head>
      <body>
        <h1>変換先を選択してください</h1>
        <ChoiceButtons />
        <Scripts />
      </body>
    </html>
  );
}
