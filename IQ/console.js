/*
 * console.js
 * To create a visual console for JavaScript testing, displays all console events. (Also great for game output)
 * Written by Deepseek V3 AI Feb 10th 2025
 * Inspired, co-authored and modified by ceneezer (see bellow comments)
 * Distributed under Creative Commons Licence, modify as you need, and please give public credit to Deepseek and ceneezer.
*/

DEBUG=false;
class DebugConsole {
    constructor() {
        this.initConsoleProxy();
        this.createConsoleUI();
        this.registerGlobalErrorHandler();
    }

    initConsoleProxy() {
        this.originalConsole = { ...console };

        const methods = ['log'];//, 'error', 'warn', 'info', 'debug'];
        methods.forEach(method => {
            console[method] = (...args) => {
                this.originalConsole[method](...args);
                this.processLog(method, args);
            };
        });
    }

    createConsoleUI() {
        this.panel = document.createElement('div');
        this.panel.id = 'debug-console';

        const header = document.createElement('div');
        header.className = 'console-header';

        this.toggleButton = document.createElement('button');
        this.toggleButton.textContent = '▼';
        this.toggleButton.className = 'console-toggle';

        const title = document.createElement('span');
        title.textContent = 'Lost In The Digital Roots';

        const clearButton = document.createElement('button');
        clearButton.textContent = 'Retry'; //'Clear';
        clearButton.className = 'console-clear';

        header.append(this.toggleButton, title, clearButton);

        this.content = document.createElement('div');
        this.content.className = 'console-content';

        this.panel.append(header, this.content);
        document.body.appendChild(this.panel);

        // Event listeners
        this.toggleButton.addEventListener('click', () =>
            this.panel.classList.toggle('collapsed'));

        clearButton.addEventListener('click', () =>
            document.location.reload());//this.content.innerHTML = '');
    }

    processLog(type, args) {
        const entry = document.createElement('div');
        entry.className = `log-entry log-${type}`;

        const timestamp = document.createElement('span');
        timestamp.className = 'log-timestamp';
        timestamp.textContent = new Date().toLocaleTimeString();

        const content = document.createElement('div');
        content.className = 'log-content';

        args.forEach(arg => {
            content.appendChild(this.createLogNode(arg));
        });

        entry.append(timestamp, content);
        this.content.appendChild(entry);
        this.content.scrollTop = this.content.scrollHeight;
    }

   createLogNode(data, depth = 0, seen = new WeakSet()) {
      const node = document.createElement('div');
      node.className = 'log-node';

      if (typeof data === 'object' && data !== null) {
         // Handle circular references
         if (seen.has(data)) {
               node.textContent = '[Circular]';
               return node;
         }
         seen.add(data);

         const expander = document.createElement('span');
         expander.className = 'tree-expander';
         expander.textContent = '▶';

         const preview = document.createElement('span');
         preview.className = 'object-preview';

         // Better type detection
         const typeName = data.constructor.name;
         const isArray = Array.isArray(data);
         const isSet = data instanceof Set;
         const isMap = data instanceof Map;

         preview.textContent = isArray ? `Array[${data.length}]` :
               isSet ? `Set[${data.size}]` :
               isMap ? `Map[${data.size}]` :
               `${typeName} ${JSON.stringify(data, (k, v) =>
                  typeof v === 'object' ? undefined : v, 2)}`;

         const children = document.createElement('div');
         children.className = 'tree-children';

         expander.addEventListener('click', () => {
               if (!children.hasChildNodes()) {
                  try {
                     // Handle different collection types
                     const entries = isArray ? data.entries() :
                           isSet ? Array.from(data).map((v, i) => [i, v]) :
                           isMap ? data.entries() :
                           Object.entries(data);

                     for (const [key, value] of entries) {
                           const childNode = document.createElement('div');
                           childNode.className = 'tree-child';

                           const keySpan = document.createElement('span');
                           keySpan.className = 'tree-key';
                           keySpan.textContent = isArray ? `[${key}]` :
                                             isSet ? `[entry ${key}]` :
                                             `${key}:`;

                           childNode.append(
                              keySpan,
                              this.createLogNode(value, depth + 1, seen)
                           );
                           children.appendChild(childNode);
                     }
                  } catch (e) {
                     const errorNode = document.createElement('div');
                     errorNode.className = 'log-error';
                     errorNode.textContent = `[Error: ${e.message}]`;
                     children.appendChild(errorNode);
                  }
               }
               children.classList.toggle('visible');
               expander.textContent = children.classList.contains('visible') ?
                  '▼' : '▶';
         });

         node.append(expander, preview, children);
      } else {
         node.textContent = this.formatPrimitive(data);
         node.className = `log-primitive ${typeof data}`;
      }

      return node;
   }

    formatPrimitive(value) {
        if (typeof value === 'string') return `"${value}"`;
        if (value === undefined) return 'undefined';
        return value.toString();
    }

    registerGlobalErrorHandler() {
        window.addEventListener('error', (e) => {
            console.error('Uncaught Error:', e.error);
        });

        window.addEventListener('unhandledrejection', (e) => {
            console.error('Unhandled Promise Rejection:', e.reason);
        });
    }
}

// Initialize console when DOM is ready
if (DEBUG===false)
document.addEventListener('DOMContentLoaded', () => {
    new DebugConsole();
    console.log("I don't track you, use cookies, advertise or even record scores!");
    console.log("I only hope you enjoy... and maybe learn a little about digital roots.");
    console.log("For a soundtrack I recommend: http://ceneezer.icu/midi.htm");
});
