/*
 * controls.js
 * To process player input in "Lost In The Digital Roots" a game by ceneezer
 * Written by Deepseek V3 AI Feb 10th 2025
 * Inspired, co-authored and modified by ceneezer (see bellow comments)
 * Distributed under Creative Commons Licence, modify as you need, and please give public credit to Deepseek and ceneezer.
*/

const canvas = document.getElementById('cityCanvas');
const movementKeys = {
   'KeyW': { dx: 0, dy: -1 },
   'ArrowUp': { dx: 0, dy: -1 },
   'KeyS': { dx: 0, dy: 1 },
   'ArrowDown': { dx: 0, dy: 1 },
   'KeyA': { dx: -1, dy: 0 },
   'ArrowLeft': { dx: -1, dy: 0 },
   'KeyD': { dx: 1, dy: 0 },
   'ArrowRight': { dx: 1, dy: 0 }
};

document.addEventListener('keydown', (event) => {
   if (movementKeys[event.code]) {
      event.preventDefault();
      const direction = movementKeys[event.code];
      movecount++;
      handleMovement(direction.dx, direction.dy);
   } else if (event.code === 'Space') {
      event.preventDefault();
      handleRandomHop();
   }
});

function handleRandomHop() {
   if (!game.map || !game.player) return;

   // Generate all possible directions including diagonals
   const directions = [
      { dx: 0, dy: -2 },   // N
      { dx: -2, dy: 0 },   // W
      { dx: 2, dy: 0 },   // E
      { dx: 0, dy: 2 },   // S
   ];

   // Filter valid moves first
   const validMoves = directions.filter(({ dx, dy }) => {
      const testX = (game.player.x + dx);
      const testY = (game.player.y + dy);
      if (testX>=game.map.size || testX<0)
         return false;
      if (testY>=game.map.size || testY<0)
         return false;
      if (!game.map.rooms[testY] || !game.map.rooms[testY][testX])
         return false;
      return game.map.rooms[testY][testX].path;
   });

   if (validMoves.length > 0) {
      // Pick random valid direction
      const randomDir = validMoves[Math.floor(Math.random() * validMoves.length)];
      handleMovement(randomDir.dx, randomDir.dy);

      // Add visual feedback
      flashPlayer();
   } else {
      console.log("No valid moves available!");
   }
}

// Add visual feedback animation
let isFlashing = false; // Track animation state

function flashPlayer() {
   if (isFlashing || !game.player?.image?.ctx) return;

   isFlashing = true;
   const playerCanvas = game.player.image;
   const ctx = playerCanvas.ctx;
   const originalFill = ctx.fillStyle;

   // Store original sprite
   const originalSprite = ctx.getImageData(0, 0, 32, 32);

   // Flash effect
   ctx.fillStyle = '#FF00FF';
   ctx.globalAlpha = 0.7;
   ctx.fillRect(0, 0, 32, 32);

   setTimeout(() => {
      // Restore original sprite
      ctx.putImageData(originalSprite, 0, 0);
      isFlashing = false;

      // Force redraw of entire scene
      const mainCanvas = document.getElementById('cityCanvas');
      const mainCtx = mainCanvas.getContext('2d');
      mainCtx.clearRect(0, 0, mainCanvas.width, mainCanvas.height);
      drawCity('cityCanvas',
         game.map.rooms.map(row => row.map(room => room.value)),
         game.map.colorMap
      );
   }, 100);
}

function handleMovement(dx, dy) {
   if (!game.map || !game.player) return;

   const size = game.map.size;
   const currentX = game.player.x;
   const currentY = game.player.y;

   // Calculate new position with wrap-around
   const newX = (currentX + dx + size) % size;
   const newY = (currentY + dy + size) % size;

   // Get target room reference
   const targetRoom = game.map.rooms[newY][newX];

   if (targetRoom.path) {
      // Update player position
      game.player.x = newX;
      game.player.y = newY;

      // Redraw scene
      drawCity('cityCanvas',
         game.map.rooms.map(row => row.map(room => room.value)),
         game.map.colorMap
      );

      // Process room events
      processRoomEvent(newX, newY);
   }
}

est=2^2;

function handleBaseProgression() {
   if (!game.player || !game.map) return;
   IQ=1;
   switch (game.map.base) {
      case 3:
         IQ=4/movecount*10+90;
         break;
      case 4:
         IQ=6/movecount*15+90;
         break;
      case 5:
         IQ=7/movecount*20+90;
         break;
      case 6:
         IQ=15/movecount*25+90;
         break;
      case 7:
         IQ=28/movecount*20+100;
         break;
      case 8:
         IQ=57/movecount*25+100;
         break;
      case 9:
         IQ=98/movecount*25+110;
         break;
      case 10:
         IQ=176/movecount*25+120;
         break;
//if you got this far without knowing the patterns, your a genius - 250 moves by my estimate should be nearly impossible.
      default:
         if (game.map.base==11) //262 - but missed 1hp
            est=176;
         if (game.map.base>10) {
            est+=(game.map.base-2)^2;
            IQ=est/movecount*4*game.map.base+100+game.map.base/2*5;
         }
   }
console.log("IQ estimate: >"+(Math.round(IQ*100)/100));
   if (IQ>150)
console.log("Email me with your thoughts so far! ceneezer@gmail.com");
   const newBase = game.map.base + 1;
   newSize = newBase-1; // Follow original size rule
   if (newBase==3)
      newSize=3;
   // Cap maximum base for performance
   if (newBase > 36) {
      console.log("you won!");
      return;
   }
   // Generate new city
   game.map = createMap(
      generateCity(newSize, newBase).array,
      newBase,
      'cityCanvas'
   );
   // Reset player position
   game.player.x = 1;
   game.player.y = 1;
   game.player.health.max=newBase;
   for (i=1;i<newBase-1;i++)
      for (j=1;j<newBase-1;j++) {
         if (game.map.rooms[i][j].value%newBase==7)
            createEntity('enemy', null, {
            x: i,
            y: j,
            health: { current: 1, max: 1 },
            onReady: () => {
if (DEBUG===true) console.log('Enemy sprite loaded');
            }
         });
         if (newBase>10 && game.map.rooms[i][j].value%newBase==9)
            createEntity('enemy', null, {
            x: i,
            y: j,
            health: { current: 3, max: 3 },
            onReady: () => {
if (DEBUG===true) console.log('Enemy sprite loaded');
            }
         });
         if (newBase>12 && game.map.rooms[i][j].value%newBase==11)
            createEntity('enemy', null, {
            x: i,
            y: j,
            health: { current: 5, max: 5 },
            onReady: () => {
if (DEBUG===true) console.log('Enemy sprite loaded');
            }
         });
      }
   console.log(`Advanced to base ${newBase} system!`);
   document.dispatchEvent(new CustomEvent('baseChanged', {
      detail: { base: newBase, size: newSize }
   }));

   // Redraw everything
   drawCity('cityCanvas',
      game.map.rooms.map(row => row.map(room => room.value)),
      game.map.colorMap
   );
}

ENEMY=null;
movecount=0;

function addHealth(x) {
   for (i=0; i<x; i++)
      if (game.player.health.current<game.player.health.max)
         game.player.health.current++;
console.log("Health added, totaling: "+game.player.health.current);
}

function processRoomEvent(x, y) {
   if (game.player.health.current<1)
      return;
   if (!game.player.inventory["Facts"]) {
      game.player.inventory["Facts"]=1;
      game.map.rooms[y][x].visited = true;
   }
   const roomValue = game.map.rooms[y][x].value;
if (DEBUG===true) console.log(`Entered room (${x},${y}) with value ${roomValue}`);
   roomHasEnemy=-1;
   game.enemies.forEach((row, i) => {
      if (row.x==x && row.y==y)
         roomHasEnemy=i;
   });
//Bosses chase
   if (roomValue==game.map.base-1 && roomValue>1) {
      game.enemies[0].x=x;
      game.enemies[0].y=y;
      roomHasEnemy=0;
   }
   if (roomHasEnemy>-1) {
      if (engageCombat(game, roomHasEnemy) === game.player) {
         console.log("Enemy defeated!");
      } else {
         console.log("OH NO! the enemy got you!   Highest level: ", game.map.base, "number of moves:", movecount);
         return;
      }
   }
//Empty city?
   cityHP=0;
   game.enemies.forEach((row, i) => {
      cityHP+=row.health.current;
   });
if (DEBUG===true) console.log("City HP left:"+cityHP);
   if (cityHP==0 && game.map.base>2) {
      ENEMY.health.max=game.map.base-1;
      ENEMY.health.current=ENEMY.health.max;
      ENEMY.x=0;
      ENEMY.y=0;
      x=1;
      y=1;
      handleBaseProgression();
      game.player.inventory["Facts"]++;
   }
// Add game logic for room interactions here
   if (!game.map.rooms[y][x].visited) {
      game.map.rooms[y][x].visited = true;
      if (roomValue%4==2)
         game.map.rooms[y][x].path = false; //one hit rooms
      if (roomValue%4==0)
         addHealth(1);
      if (roomValue%10==0)
         addHealth(3);
//      if (roomValue%10==5)
//         addHealth(3); (unsure what to add here... DoT? ranged? tech tree?)
      if (roomValue%13==0)
         addHealth(5);
      if (roomValue%2==1) { //progress
         game.player.inventory["Facts"]++;
console.log("Facts: "+game.player.inventory["Facts"]
   ,"Move counter: "+movecount);
         if (game.player.inventory["Facts"]==61) {
            game.player.health.current++;
console.log("BONUS +1HP awarded - good thing you've hit every odd square in so far!");
         }
         if (game.player.inventory["Facts"]==108) {
            game.player.health.current+=2;
console.log("BONUS +2HP awarded, total:"+game.player.health.current);
         }
         if (game.player.inventory["Facts"]==134) {
            game.player.health.current+=2;
console.log("BONUS +2HP awarded, total:"+game.player.health.current);
         }
      }
   }
//Base 2
   if (game.map.base==2 && game.player.inventory["Facts"]>8) {
console.log("IQ estimate: >"+(3/movecount*40+80));
      ENEMY=createEntity('enemy', null, {
         x: 0,
         y: 0,
         health: {current: 1, max: game.map.base-1},
         onReady: () => {
if (DEBUG===true) console.log('Enemy sprite loaded');
      }});
      handleBaseProgression();
   }
// base 3: 18XP // -2 8HP remain
// base 4: 20XP // -3 5HP remain
// base 5: 25XP // -4 (+1 after, square 4 battle) 2HP remain
// base 6: 33XP // +4 -5 (1hp remain hereafter - boss *requires full health)
// base 7: 43XP // +6 -6
// base 8: 61XP // +6(+1XP) -7 (cannot complete without hitting every odd square before here)
// base 9: 78XP // +12 -8 4x7s - 141 moves till here minimum (not using hops better :D  )
// base 10: 108XP // +12(+2XP) -9 -6x7s -
// base 11: 134XP // +24(+2XP) -10 -4x7s -4x10s
// base 12: 186XP // +50 -10x4hp-11hp -10x7s -10x10s (missing 3 extras?)
// Base 13: >218XP // +20+54 -34-12 -4x7s -10x10s
// Base 14: 290XP // +12+36 -36hp -12x7s -12x10s
   drawCity('cityCanvas',
      game.map.rooms.map(row => row.map(room => room.value)),
      game.map.colorMap);
}
//DEBUG=true;

// Initialize controls with current game state
if (DEBUG===true) console.log("Movement controls initialized");
