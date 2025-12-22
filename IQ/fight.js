/*
 * fight.js
 * To process combat in "Lost In The Digital Roots" a game by ceneezer
 * Written by Deepseek V3 AI Feb 10th 2025
 * Inspired, co-authored and modified by ceneezer (see bellow comments)
 * Distributed under Creative Commons Licence, modify as you need, and please give public credit to Deepseek and ceneezer.
*/

function engageCombat(game, i) {
    // Initialize strengths
    player=game.player;
    player.strength = player.strength || 1;

    // Get max room value from current city
    const maxRoomValue = Math.max(...game.map.rooms.flat().map(room =>
        parseInt(room.value, 10)
    ));
    enemy=game.enemies[i] || {x:0,y:0,health:maxRoomValue};
    enemy.strength = enemy.strength || 1;

if (DEBUG===true) console.log("IN COMABAT!!");
    // Empower enemy if on max-value room
    const enemyRoomValue = parseInt(game.map.rooms[enemy.y][enemy.x].value, 10);
    if (enemyRoomValue === maxRoomValue) {
        enemy.health.current = maxRoomValue;
        console.log(`Enemy supercharged by ${maxRoomValue} energy!`);
    }

    console.log(`⚔️ Combat Start ⚔️
    Player: ${player.health.current}HP/${player.health.max}HP
    Enemy: ${enemy.health.current}HP`);

    let round = 1;
    while (player.health.current > 0 && enemy.health.current > 0) {
        console.log(`\nRound ${round} ------------------`);

        // Enemy attack phase
        player.health.current -= enemy.strength;
        console.log(`🛡️  Enemy strikes for ${enemy.strength} damage!
        Player HP: ${Math.max(player.health.current, 0)}`);

        if (player.health.current <= 0) break;

        // Player attack phase
        enemy.health.current -= player.strength;
        console.log(`⚡ Player retaliates for ${player.strength} damage!
        Enemy HP: ${Math.max(enemy.health.current, 0)}`);

        round++;
    }

    const victor = player.health.current > 0 ? player : enemy;
    if (victor==player) console.log(`🏆 Victory!`);
    console.log(`${victor.type} remains standing!`);
    if (i>0 && enemy.health.current<1)
       delete game.enemies[i];
    return victor;
}
