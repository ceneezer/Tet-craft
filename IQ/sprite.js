/*
 * sprite.js
 * To generate graphics for "Lost In The Digital Roots" a game by ceneezer
 * Written by Deepseek V3 AI Feb 10th 2025
 * Inspired, co-authored and modified by ceneezer (see bellow comments)
 * Distributed under Creative Commons Licence, modify as you need, and please give public credit to Deepseek and ceneezer.
*/

// Global game objects
const game = {
    map: null,
    player: null,
    enemies: []
};

// Initialize rooms and color map
function initializeRooms(city) {
    // Add array validation
    if (!Array.isArray(city)) {
        console.error('Invalid city data:', city);
        return [[]]; // Fallback empty grid
    }

    return city.map(row => {
        // Validate row structure
        if (!Array.isArray(row)) {
            console.warn('Invalid city row:', row);
            return [];
        }

        return row.map(value => ({
            visited: false,
            value: value,
            path: parseInt(value, 10) !== 0
        }));
    });
}

function createMap(cityArray, base, canvasId) {
    // Add input validation
    if (!Array.isArray(cityArray) || cityArray.some(row => !Array.isArray(row))) {
        console.error('Invalid cityArray format - regenerating default');
        cityArray = generateCity(10, 2); // Fallback to known good values
    }

    const rooms = initializeRooms(cityArray);
    const colorMap = createColorMap(cityArray, base);
    drawCity(canvasId, cityArray, colorMap);

    game.map = {
        rooms: rooms,
        colorMap: colorMap,
        base: base,
        size: cityArray.length
    };
if (DEBUG===true) console.log('Map created:', game.map);
    return game.map;
}

function createColorMap(city, base) {
    const maxValue = base - 1;
    const colorMap = {};
    const middleValues = base-2;
    const hueStep = middleValues > 0 ? 360 / middleValues : 0;

    colorMap[0] = '#000000';
    for (i=maxValue; i>0; i--) {
         const idx = i;
         const hue = 360-(hueStep * idx);
         const lightness = i % 2 === 0 ? '30%' : '70%';
         colorMap[i] = `hsl(${hue}, 100%, ${lightness})`;
    }
    colorMap[maxValue] = '#FFFFFF'
if (DEBUG===true) console.log("Colormap:",colorMap);

    return colorMap;
}

function drawCity(canvasId, city, colorMap) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    const cellSize = Math.min(canvas.width, canvas.height) / city.length;

    city.forEach((row, i) => {
        row.forEach((val, j) => {
            ctx.fillStyle = colorMap[parseInt(val, 10)];
            ctx.fillRect(j*cellSize, i*cellSize, cellSize, cellSize);
        });
    });
    drawEntities(ctx, cellSize);
}

// Entity creation system
function createEntity(type, imageSpec, attributes) {
    const entity = {
        type: type,
        x: 0,
        y: 0,
        statusEffects: [],
        bonuses: [],
        health: { current: 1, max: 10 },
        inventory: [],
        strength: 1,
        skills: [],
        image: generateSprite(type)
    };

    Object.assign(entity, attributes);

    if (type === 'player') {
        game.player = entity;
    } else if (type === 'enemy') {
        game.enemies.push(entity);
    }

if (DEBUG===true)  console.log(`Created ${type}:`, entity);
    return entity;
}

function drawEntities(ctx, cellSize) {
    // Draw player
    if (game.player) {
        ctx.drawImage(
            game.player.image,
            game.player.x * cellSize + cellSize/4,
            game.player.y * cellSize + cellSize/4,
            cellSize/2,
            cellSize/2
        );
    }

    // Draw enemies
    game.enemies.forEach(enemy => {
        ctx.drawImage(
            enemy.image,
            enemy.x * cellSize + cellSize/4,
            enemy.y * cellSize + cellSize/4,
            cellSize/2,
            cellSize/2
        );
    });
}

function generateSprite(type) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 32;
    canvas.height = 32;
    canvas.ctx = ctx;

    if (type === 'player') {
        ctx.fillStyle = '#000000';
        ctx.beginPath();
        ctx.moveTo(16, 0);
        ctx.lineTo(32, 32);
        ctx.lineTo(0, 32);
        ctx.closePath();
        ctx.fill();
if (DEBUG===true) console.log("Drew player", ctx);
    } else {
        ctx.fillStyle = '#000000';
        ctx.beginPath();
        ctx.arc(16, 16, 14, 0, Math.PI*2);
        ctx.fill();
if (DEBUG===true) console.log("Drew enemy", ctx);
    }
    return canvas;
}
